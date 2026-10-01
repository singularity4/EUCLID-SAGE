#!/usr/bin/env python3
r"""
build_vis_benchmarks.py - image stamps and tables of the six black-and-white (VIS) benchmarks

Part of EUCLID-SAGE: Strong Gravitational Lens Benchmark Suite for Euclid.

Benchmarks produced (choose the types with --types; default R and S)
    EUCLID-SAGE-1k-R_VIS    487 lenses +   513 randomly drawn non-lenses
    EUCLID-SAGE-1k-S_VIS    487 lenses +   513 brightness-matched non-lenses
    EUCLID-SAGE-1k-D_VIS    487 lenses +   513 non-lenses matched in brightness, Sersic index, radius, and ellipticity
    EUCLID-SAGE-10k-R_VIS   487 lenses + 9,513 randomly drawn non-lenses
    EUCLID-SAGE-10k-S_VIS   487 lenses + 9,513 brightness-matched non-lenses
    EUCLID-SAGE-10k-D_VIS   487 lenses + 9,513 non-lenses matched in brightness, Sersic index, radius, and ellipticity
    (the number of lenses is taken from the selection; the 1k non-lenses are the first entries
    of the corresponding 10k list)

Input
    nonlens_selection/lenses.csv, nonlenses_R.csv, nonlenses_S.csv   written by select_nonlenses.py
    nonlens_selection/nonlenses_D.csv                                written by select_nonlenses_D.py
    q1_discovery_engine_lens_catalog.csv                             lens positions and tiles

Method
    For each object, the ESA Euclid Science Archive is searched for the VIS background subtracted 
    MER mosaics that cover its position (table q1.mosaic_product). A region of 8 arcsec radius around 
    the catalogue position is downloaded, and a stamp of 101 x 101 pixels (0.1 arcsec per pixel, about 10 arcsec) 
    is cut out, centred on the pixel that contains the catalogue position (world coordinate system of the mosaic). 
    Pixel values are kept in the unit of the mosaic (ADU/s) and stored as 32-bit floating-point numbers.
    
    The MER tiles overlap, so objects near a tile edge are covered by two or more mosaics. The
    covering mosaics are tried in turn (for lenses the lens's own tile first, otherwise in order of
    tile index) until one yields a complete stamp, i.e. a stamp without pixels outside the mosaic.
    Non-lenses without a complete stamp are replaced by the next object of the selection order
    (reserves). Lenses cannot be replaced; a lens without a complete stamp is reported.

Options
    --types R S D            benchmark types to build (default R S); stamps already saved are reused,
                             so building D later downloads only the new D non-lenses
    --parallel-downloads N   number of simultaneous downloads (default 4; about 2 hours in total)
    --keep-downloads         keep the downloaded 16 arcsec regions (default: deleted after the stamp
                             is cut, which saves about 2 GB of disk space)
    A download that fails for a technical reason (for example a network interruption) is repeated
    once after 5 seconds.

Restarting
    Stamps already saved, and objects already rejected (listed in download_log.csv), are not
    downloaded again, so the script can be interrupted and restarted. To attempt previously rejected
    objects again, delete download_log.csv; saved stamps are still reused.

Output (Euclid/benchmarks_vis/)
    stamps/<identifier>_VIS_101.fits   one stamp per object (lenses: id_str; non-lenses: object_id)
    EUCLID-SAGE-<size>-<R|D|S>_VIS.csv one row per stamp: stamp file, label (1 lens, 0 non-lens),
                                       object identifier, lens grade, Euclid classification label,
                                       VIS magnitude, position, selection rank, matched lens (S/D)
    download_log.csv                   outcome for every object attempted
    build_report.txt                   the printed progress and summary

Run from the data folder, after select_nonlenses.py:
    python3 build_vis_benchmarks.py
    python3 build_vis_benchmarks.py --parallel-downloads 4 --keep-downloads
    python3 build_vis_benchmarks.py --types D
"""
import os
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Working folder holding the input catalogues and the outputs: the folder given in the environment
# variable EUCLID_SAGE_DATA_DIR, otherwise the folder the script is run from.
EUCLID_FOLDER = Path(os.environ.get("EUCLID_SAGE_DATA_DIR", ".")).resolve()
SELECTION_FOLDER = EUCLID_FOLDER / "nonlens_selection"
LENS_CATALOGUE = EUCLID_FOLDER / "q1_discovery_engine_lens_catalog.csv"
OUTPUT_FOLDER = EUCLID_FOLDER / "benchmarks_vis"
STAMP_FOLDER = OUTPUT_FOLDER / "stamps"
DOWNLOAD_FOLDER = OUTPUT_FOLDER / "raw"
DOWNLOAD_LOG_FILE = OUTPUT_FOLDER / "download_log.csv"

STAMP_SIZE_PIXELS = 101                  # 101 x 101 pixels, about 10 arcsec
DOWNLOAD_RADIUS_ARCSEC = 8.0             # about 16 arcsec downloaded around each object
MOSAIC_SEARCH_RADIUS_ARCSEC = 10.0       # finds every mosaic overlapping the downloaded region
VIS_PIXEL_UNIT = "ADU/s"                 # Euclid Q1 Data Product Description, VIS MER mosaics
BENCHMARK_SIZES = {"1k": 1000, "10k": 10000}
WAIT_BEFORE_REPEAT_SECONDS = 5


# ---------------------------------------------------------------- one stamp (runs in a separate process)

def find_covering_vis_mosaics(euclid_archive, right_ascension, declination, preferred_tile_index=None):
    """All VIS background-subtracted MER mosaics covering the position, as (full file path, tile
    index), with the preferred tile (the lens's own tile) first and the others in order of tile index."""
    search_radius_degrees = MOSAIC_SEARCH_RADIUS_ARCSEC / 3600
    catalogue_request = ("SELECT file_name, file_path, tile_index FROM q1.mosaic_product "
                         "WHERE instrument_name='VIS' AND "
                         f"INTERSECTS(CIRCLE({right_ascension}, {declination}, {search_radius_degrees}), fov)=1")
    mosaics = euclid_archive.launch_job(catalogue_request).get_results().to_pandas()
    for text_column in ("file_name", "file_path"):   # the archive may return text as bytes
        mosaics[text_column] = mosaics[text_column].apply(
            lambda value: value.decode() if isinstance(value, bytes) else str(value))
    mosaics = mosaics[mosaics.file_name.str.contains("BGSUB-MOSAIC-VIS")].copy()
    mosaics["is_preferred_tile"] = mosaics.tile_index.astype(str) == str(preferred_tile_index)
    mosaics = mosaics.sort_values(["is_preferred_tile", "tile_index"], ascending=[False, True])
    return [(f"{mosaic.file_path}/{mosaic.file_name}", mosaic.tile_index) for mosaic in mosaics.itertuples()]


def make_stamp(download_request):
    """Obtain and save one stamp; returns one row of the download log.
    Runs in a separate process, so all archive and image libraries are imported here."""
    import contextlib
    import io
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from astroquery import log as archive_messages
    from astroquery.esa.euclid import Euclid as euclid_archive
    archive_messages.setLevel("CRITICAL")   # long archive error pages are summarised in the log instead

    stamp_path = STAMP_FOLDER / f"{download_request['stamp_id']}_VIS_101.fits"
    outcome = {"stamp_id": download_request["stamp_id"], "object_type": download_request["object_type"]}
    if stamp_path.exists():
        outcome.update(status="ok", note="already saved")
        return outcome

    start_time = time.perf_counter()
    sky_position = SkyCoord(download_request["right_ascension"], download_request["declination"], unit="deg")
    technical_problem_text = None
    for attempt in (1, 2):   # the second attempt only after a technical failure
        try:
            suppressed_output = io.StringIO()
            with contextlib.redirect_stdout(suppressed_output), contextlib.redirect_stderr(suppressed_output):
                covering_mosaics = find_covering_vis_mosaics(
                    euclid_archive, download_request["right_ascension"], download_request["declination"],
                    download_request.get("preferred_tile_index"))
            if not covering_mosaics:
                outcome.update(status="rejected: no VIS mosaic at this position",
                               seconds=round(time.perf_counter() - start_time, 1))
                return outcome
            mosaic_problems, technical_problem = [], False
            for mosaic_file_path, tile_index in covering_mosaics:
                download_path = DOWNLOAD_FOLDER / f"{download_request['stamp_id']}_{int(tile_index)}_VIS_raw.fits"
                try:
                    with contextlib.redirect_stdout(suppressed_output), contextlib.redirect_stderr(suppressed_output):
                        saved_paths = euclid_archive.get_cutout(
                            file_path=mosaic_file_path, instrument="VIS", id=str(tile_index),
                            coordinate=sky_position, radius=DOWNLOAD_RADIUS_ARCSEC * u.arcsec,
                            output_file=str(download_path))
                    download_path = Path(saved_paths[0] if isinstance(saved_paths, (list, tuple)) else saved_paths)
                    with fits.open(download_path) as downloaded_file:
                        image_extension = next(extension for extension in downloaded_file
                                               if extension.data is not None and extension.data.ndim == 2)
                        image = image_extension.data.astype(np.float32)
                        mosaic_header = image_extension.header.copy()
                except Exception as error:
                    error_text = str(error)
                    if "not within image" in error_text:
                        mosaic_problems.append(f"tile {int(tile_index)}: outside mosaic")
                    else:
                        mosaic_problems.append(f"tile {int(tile_index)}: {error_text[:120]}")
                        technical_problem = True
                    continue
                if not download_request["keep_downloads"]:
                    download_path.unlink(missing_ok=True)
                # The stamp is centred on the pixel that contains the catalogue position (within
                # 0.05 arcsec). Pixels outside the mosaic are marked as missing (NaN).
                stamp = Cutout2D(image, sky_position, (STAMP_SIZE_PIXELS, STAMP_SIZE_PIXELS),
                                 wcs=WCS(mosaic_header), mode="partial", fill_value=np.nan)
                missing_fraction = float(np.isnan(stamp.data).mean())
                if missing_fraction > 0:
                    mosaic_problems.append(f"tile {int(tile_index)}: missing pixels ({missing_fraction:.3f})")
                    continue
                stamp_header = stamp.wcs.to_header()
                stamp_header["BUNIT"] = (VIS_PIXEL_UNIT, "pixel unit of the VIS MER mosaic")
                stamp_header["MAGZERO"] = (mosaic_header.get("MAGZERO"), "AB zero point of the VIS mosaic")
                stamp_header["TILEIDX"] = (int(tile_index), "MER tile index of the source mosaic")
                stamp_header["MOSAIC"] = Path(mosaic_file_path).name   # long values continue on further lines
                stamp_header["OBJTYPE"] = (download_request["object_type"], "lens or nonlens")
                for keyword, (value, description) in download_request["extra_header_entries"].items():
                    stamp_header[keyword] = (value, description)
                fits.writeto(stamp_path, stamp.data, stamp_header, overwrite=True)
                outcome.update(status="ok", tile_index=int(tile_index), mosaic_file=Path(mosaic_file_path).name,
                               mosaics_covering=len(covering_mosaics), note="; ".join(mosaic_problems),
                               seconds=round(time.perf_counter() - start_time, 1))
                return outcome
            if technical_problem and attempt == 1:
                technical_problem_text = "; ".join(mosaic_problems)
                time.sleep(WAIT_BEFORE_REPEAT_SECONDS)
                continue
            # No covering mosaic yields a complete stamp: non-lenses are replaced, lenses reported.
            outcome.update(status="rejected: no complete stamp in any covering mosaic",
                           mosaics_covering=len(covering_mosaics), note="; ".join(mosaic_problems),
                           seconds=round(time.perf_counter() - start_time, 1))
            return outcome
        except Exception as error:
            technical_problem_text = str(error)[:200]
            if attempt == 1:
                time.sleep(WAIT_BEFORE_REPEAT_SECONDS)
    outcome.update(status=f"rejected: download failed ({technical_problem_text})",
                   seconds=round(time.perf_counter() - start_time, 1))
    return outcome


# ---------------------------------------------------------------- coordination of all stamps

class ProgressReport:
    """Prints each line and keeps a copy for build_report.txt."""
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)


def make_stamps(download_requests, parallel_downloads, download_records, progress_report, description):
    """Make the stamps of all requests in parallel, skipping objects rejected in an earlier run.
    Returns the outcomes by stamp identifier."""
    earlier_rejections = {record["stamp_id"]: record for record in download_records
                          if record["status"].startswith("rejected")}
    requests_to_run = [request for request in download_requests if request["stamp_id"] not in earlier_rejections]
    outcomes = [earlier_rejections[request["stamp_id"]] for request in download_requests
                if request["stamp_id"] in earlier_rejections]
    progress_report(f"{description}: {len(download_requests):,} objects "
                    f"({len(download_requests) - len(requests_to_run):,} already rejected earlier)")
    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=parallel_downloads) as parallel_processes:
        pending_stamps = [parallel_processes.submit(make_stamp, request) for request in requests_to_run]
        for number_finished, finished_stamp in enumerate(as_completed(pending_stamps), start=1):
            outcome = finished_stamp.result()
            outcomes.append(outcome)
            download_records.append(outcome)
            if number_finished % 200 == 0 or number_finished == len(pending_stamps):
                elapsed_minutes = (time.perf_counter() - start_time) / 60
                progress_report(f"  {number_finished:,}/{len(pending_stamps):,} done, {elapsed_minutes:.0f} min")
                pd.DataFrame(download_records).drop_duplicates("stamp_id", keep="last").to_csv(
                    DOWNLOAD_LOG_FILE, index=False)
    return {outcome["stamp_id"]: outcome for outcome in outcomes}


def nonlens_download_request(nonlens, keep_downloads):
    """Download request for one non-lens (a row of nonlenses_R.csv or nonlenses_S.csv)."""
    return {"object_type": "nonlens", "stamp_id": str(int(nonlens.object_id)),
            "right_ascension": nonlens.right_ascension, "declination": nonlens.declination,
            "preferred_tile_index": None, "keep_downloads": keep_downloads,
            "extra_header_entries": {
                "OBJECTID": (int(nonlens.object_id), "Euclid MER object_id"),
                "PHZLABEL": (int(nonlens.label), "phz_classification: +1 star, +2 galaxy, +4 QSO")}}


def main():
    argument_parser = argparse.ArgumentParser(
        description="Download the VIS stamps and assemble the EUCLID-SAGE black-and-white benchmarks")
    argument_parser.add_argument("--types", nargs="+", default=["R", "S"], choices=["R", "S", "D"],
                                 help="benchmark types to build (default R S)")
    argument_parser.add_argument("--parallel-downloads", type=int, default=4,
                                 help="number of simultaneous downloads (default 4)")
    argument_parser.add_argument("--keep-downloads", action="store_true",
                                 help="keep the downloaded 16 arcsec regions")
    settings = argument_parser.parse_args()
    STAMP_FOLDER.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    progress_report = ProgressReport()
    download_records = pd.read_csv(DOWNLOAD_LOG_FILE).to_dict("records") if DOWNLOAD_LOG_FILE.exists() else []
    for record in download_records:
        record["stamp_id"] = str(record["stamp_id"])

    # ---- Input ----
    lenses = pd.read_csv(SELECTION_FOLDER / "lenses.csv")
    lens_positions = pd.read_csv(LENS_CATALOGUE)[["id_str", "right_ascension", "declination", "tile_index"]]
    lenses = lenses.merge(lens_positions, on="id_str", how="left")
    lenses["vis_magnitude"] = -2.5 * np.log10(pd.to_numeric(lenses.flux_detection_total, errors="coerce")) + 23.9
    nonlens_lists = {benchmark_type: pd.read_csv(SELECTION_FOLDER / f"nonlenses_{benchmark_type}.csv")
                     for benchmark_type in settings.types}
    number_of_lenses = len(lenses)
    nonlenses_needed = {size_name: size - number_of_lenses for size_name, size in BENCHMARK_SIZES.items()}
    progress_report(f"Lenses: {number_of_lenses}; non-lenses needed per benchmark: {nonlenses_needed}")

    # ---- 1. Lens stamps ----
    lens_download_requests = [
        {"object_type": "lens", "stamp_id": lens.id_str,
         "right_ascension": lens.right_ascension, "declination": lens.declination,
         "preferred_tile_index": int(lens.tile_index) if pd.notna(lens.tile_index) else None,
         "keep_downloads": settings.keep_downloads,
         "extra_header_entries": {"LENSID": (lens.id_str, "lens catalogue id_str"),
                                  "GRADE": (lens.grade, "expert grade of the lens candidate"),
                                  "PHZLABEL": (int(lens.phz_classification), "phz_classification")}}
        for lens in lenses.itertuples()]
    lens_outcomes = make_stamps(lens_download_requests, settings.parallel_downloads, download_records,
                                progress_report, "Lens stamps")
    lenses_without_stamp = [stamp_id for stamp_id, outcome in lens_outcomes.items() if outcome["status"] != "ok"]
    progress_report(f"  lens stamps complete: {number_of_lenses - len(lenses_without_stamp)}; "
                    f"without stamp: {lenses_without_stamp or 'none'}")

    # ---- 2. Non-lens stamps of the 10k lists (all chosen types together; shared objects only once) ----
    nonlenses_in_10k = pd.concat([nonlens_list[nonlens_list.in_10k] for nonlens_list in nonlens_lists.values()])
    nonlenses_in_10k = nonlenses_in_10k.drop_duplicates("object_id")
    nonlens_outcomes = make_stamps(
        [nonlens_download_request(nonlens, settings.keep_downloads) for nonlens in nonlenses_in_10k.itertuples()],
        settings.parallel_downloads, download_records, progress_report,
        f"Non-lens stamps ({', '.join(settings.types)})")

    # ---- 3. Replacement of rejected non-lenses by reserves, in selection order ----
    for benchmark_type, nonlens_list in nonlens_lists.items():
        while True:
            attempted_identifiers = set(nonlens_outcomes)
            attempted = nonlens_list[nonlens_list.object_id.astype(str).isin(attempted_identifiers)]
            accepted = attempted[[nonlens_outcomes[str(identifier)]["status"] == "ok"
                                  for identifier in attempted.object_id]]
            shortfall = nonlenses_needed["10k"] - len(accepted)
            not_yet_attempted = nonlens_list[~nonlens_list.object_id.astype(str).isin(attempted_identifiers)]
            if shortfall <= 0 or not_yet_attempted.empty:
                break
            next_reserves = not_yet_attempted.head(shortfall)
            nonlens_outcomes.update(make_stamps(
                [nonlens_download_request(nonlens, settings.keep_downloads) for nonlens in next_reserves.itertuples()],
                settings.parallel_downloads, download_records, progress_report, f"Reserves for {benchmark_type}"))
        progress_report(f"  {benchmark_type}: accepted {min(len(accepted), nonlenses_needed['10k']):,} of "
                        f"{nonlenses_needed['10k']:,} needed; rejected {len(attempted) - len(accepted)} "
                        "(replaced from reserves)"
                        + ("" if shortfall <= 0 else f"; SHORT BY {shortfall} (reserves exhausted)"))
        nonlens_lists[benchmark_type] = accepted.sort_values("rank")

    # ---- 4. Benchmark tables ----
    lens_rows = pd.DataFrame({
        "stamp_file": [f"stamps/{lens_id}_VIS_101.fits" for lens_id in lenses.id_str], "label": 1,
        "object_id": lenses.object_id, "lens_id": lenses.id_str, "grade": lenses.grade,
        "phz_label": lenses.phz_classification, "vis_magnitude": lenses.vis_magnitude.round(4),
        "right_ascension": lenses.right_ascension, "declination": lenses.declination})
    lens_rows = lens_rows[~lens_rows.lens_id.isin(lenses_without_stamp)]
    for benchmark_type, accepted_nonlenses in nonlens_lists.items():
        for size_name in BENCHMARK_SIZES:
            benchmark_nonlenses = accepted_nonlenses.head(nonlenses_needed[size_name])
            nonlens_rows = pd.DataFrame({
                "stamp_file": [f"stamps/{int(identifier)}_VIS_101.fits" for identifier in benchmark_nonlenses.object_id],
                "label": 0, "object_id": benchmark_nonlenses.object_id, "lens_id": "", "grade": "",
                "phz_label": benchmark_nonlenses.label, "vis_magnitude": benchmark_nonlenses.magnitude.round(4),
                "right_ascension": benchmark_nonlenses.right_ascension,
                "declination": benchmark_nonlenses.declination, "selection_rank": benchmark_nonlenses["rank"],
                "matched_lens_id": (benchmark_nonlenses["matched_lens_id"]
                                    if "matched_lens_id" in benchmark_nonlenses else "")})
            benchmark_table = pd.concat([lens_rows, nonlens_rows], ignore_index=True)
            benchmark_name = f"EUCLID-SAGE-{size_name}-{benchmark_type}_VIS"
            benchmark_table.to_csv(OUTPUT_FOLDER / f"{benchmark_name}.csv", index=False)
            progress_report(f"{benchmark_name}: {len(benchmark_table):,} stamps "
                            f"({int(benchmark_table.label.sum())} lenses, "
                            f"{int((benchmark_table.label == 0).sum()):,} non-lenses)")

    pd.DataFrame(download_records).drop_duplicates("stamp_id", keep="last").to_csv(DOWNLOAD_LOG_FILE, index=False)
    (OUTPUT_FOLDER / "build_report.txt").write_text("\n".join(progress_report.lines) + "\n")
    progress_report(f"\nBenchmarks, stamps and logs saved in {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
