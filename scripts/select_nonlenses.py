#!/usr/bin/env python3
r"""
select_nonlenses.py - selection of the lenses and non-lenses of the R and S benchmarks

Part of EUCLID-SAGE: Strong Gravitational Lens Benchmark Suite for Euclid.

Purpose
   Selects the lens sample used in all benchmarks and constructs ordered non-lens lists for the R (random) and S (similar) benchmarks, 
   including reserve objects. This script only reads values from the official Euclid Q1 catalogues. Image stamps are downloaded 
   by build_vis_benchmarks.py.

Classification label
    Euclid's phz_classification (table catalogue.phz_classification) is a sum of flags:
    +1 star, +2 galaxy, +4 quasar (Euclid Q1 Data Product Description Document).

Selection rules
    Lenses (the same 487 in R and S): every grade A or B lens candidate of the main search that
        carries a classification label from 1 to 7. This includes one lens labelled "star", which
        the lens team confirmed as a lens. Ten lenses without a label (no class, unclassified, or 
        absent from the table) are not used.
    R non-lenses: all classified objects (labels 1 to 7: star, galaxy, galaxy and star, quasar,
        and their combinations), drawn at random.
    S non-lenses: galaxies (labels 2 and 3: galaxy, galaxy and star) matched to the lenses in
        brightness only: for each lens, the unused galaxy with the closest total VIS magnitude.
    Both non-lens types: not flagged as a false detection by Euclid (spurious_flag = 0); not a lens
        candidate of any grade or subset; at least 12 arcsec from every lens candidate; random seed
        42; the 1k non-lenses are the first entries of the 10k list.

Sampling from the archive
    A fixed pseudo-random subset of the catalogue is selected using the last digits of each Euclid 
    object identifier. Because the identifier encodes sky position, these digits are effectively 
    unrelated to brightness or object type. The subset for R contains about 0.12% of all objects (last four
    digits below 12); the subset for S about 10% of the galaxies brighter than the faintest lens
    plus 0.2 magnitudes (last two digits below 10). Archive results are sorted by object identifier 
    before any random selection because the returned row order varies between requests; without sorting, 
    the same random seed would produce different samples between runs.

Reserves
    10% additional non-lenses are selected as reserves. During stamp extraction, a selected object is 
    replaced by the next reserve if no overlapping VIS mosaic yields a complete 101 × 101 stamp centred 
    on its catalogue position.    
    
Diagnostics
    Lens counts; representativeness of the R subset (its brightness distribution compared with
    counts over all Q1 galaxies from compare_brightness.py); quality of the S brightness matching;
    numbers of excluded objects.

Output (Euclid/nonlens_selection/)
    lenses.csv                         the 487 lenses (the same in R and S)
    nonlenses_R.csv, nonlenses_S.csv   ordered non-lenses: rank, object_id, position, label, flux,
                                       magnitude, in_1k, in_10k, reserve (S also matched_lens_id and
                                       magnitude_difference)
    selection_report.txt               the printed diagnostics

Run from the data folder, after lens_catalogue_properties.py:
    python3 select_nonlenses.py
"""
import os
import bisect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.esa.euclid import Euclid
from scipy.spatial import cKDTree

# Working folder holding the input catalogues and the outputs: the folder given in the environment
# variable EUCLID_SAGE_DATA_DIR, otherwise the folder the script is run from.
EUCLID_FOLDER = Path(os.environ.get("EUCLID_SAGE_DATA_DIR", ".")).resolve()
LENS_CATALOGUE = EUCLID_FOLDER / "q1_discovery_engine_lens_catalog.csv"
LENS_PROPERTIES = EUCLID_FOLDER / "lens_catalogue_properties.csv"   # written by lens_catalogue_properties.py
OUTPUT_FOLDER = EUCLID_FOLDER / "nonlens_selection"

RANDOM_SEED = 42
BENCHMARK_SIZES = {"1k": 1000, "10k": 10000}
RESERVE_FRACTION = 0.10
EXCLUSION_RADIUS_ARCSEC = 12.0
LABELS_R = (1, 2, 3, 4, 5, 6, 7)   # all classified objects
LABELS_S = (2, 3)                  # galaxy; galaxy and star
SUBSET_CONDITION_R = "MOD(ABS(merged_catalogue.object_id), 10000) < 12"   # about 0.12% of all objects
SUBSET_CONDITION_S = "MOD(ABS(merged_catalogue.object_id), 100) < 10"     # about 10% of all objects
FAINT_LIMIT_MARGIN_MAGNITUDES = 0.2
# Percentage of all Q1 galaxies (labels 2 and 3, not flagged as false detections) brighter than
# each VIS magnitude, measured with compare_brightness.py; used to check the R subset.
PERCENT_OF_Q1_GALAXIES_BRIGHTER = {19: 0.34, 20: 0.72, 21: 1.62, 22: 3.80, 23: 9.16, 24: 22.19, 25: 51.98}


class ProgressReport:
    """Prints each line and keeps a copy for selection_report.txt."""
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)


def magnitude_from_flux(flux_in_microjansky):
    """AB magnitude from the total VIS flux in microjansky (column flux_detection_total)."""
    return -2.5 * np.log10(flux_in_microjansky) + 23.9


def object_identifier_from_id_str(id_str):
    """Euclid object identifier from a lens catalogue id_str, exactly:
    '102018665_NEG570040238507752998' -> -570040238507752998. Returns None if absent."""
    try:
        return int(str(id_str).split("_", 1)[1].replace("NEG", "-"))
    except (IndexError, ValueError):
        return None


def unit_vectors(right_ascension_degrees, declination_degrees):
    """Positions on the sky as three-dimensional unit vectors, for fast angular-distance searches."""
    right_ascension, declination = np.radians(right_ascension_degrees), np.radians(declination_degrees)
    return np.c_[np.cos(declination) * np.cos(right_ascension),
                 np.cos(declination) * np.sin(right_ascension),
                 np.sin(declination)]


def read_catalogue_subset(catalogue_request, progress_report):
    """Read a catalogue subset from the archive, sorted by object identifier (the archive returns
    rows in a different order for each request; sorting makes the seeded selection reproducible)."""
    archive_request = Euclid.launch_job_async(catalogue_request)
    if archive_request is None:
        progress_report("  ERROR: the archive request failed; nothing was selected.")
        sys.exit(1)
    return archive_request.get_results().to_pandas().sort_values("object_id").reset_index(drop=True)


def remove_lens_candidates_and_neighbours(candidates, lens_catalogue):
    """Remove every lens candidate of the catalogue (any grade, any subset) and every object within
    the exclusion radius of one. Returns (remaining candidates, number removed as lens candidates,
    number removed as neighbours)."""
    lens_candidate_identifiers = {identifier for identifier in
                                  map(object_identifier_from_id_str, lens_catalogue.id_str.dropna())
                                  if identifier is not None}
    is_lens_candidate = candidates.object_id.isin(lens_candidate_identifiers)
    candidates_with_position = lens_catalogue.dropna(subset=["right_ascension", "declination"])
    sky_search_tree = cKDTree(unit_vectors(candidates.right_ascension.values, candidates.declination.values))
    # Straight-line distance between unit vectors separated by the exclusion radius on the sky.
    exclusion_chord_length = 2 * np.sin(np.radians(EXCLUSION_RADIUS_ARCSEC / 3600) / 2)
    is_near_lens_candidate = np.zeros(len(candidates), dtype=bool)
    lens_candidate_vectors = unit_vectors(candidates_with_position.right_ascension.values,
                                          candidates_with_position.declination.values)
    for nearby_rows in sky_search_tree.query_ball_point(lens_candidate_vectors, exclusion_chord_length):
        is_near_lens_candidate[nearby_rows] = True
    is_near_lens_candidate = pd.Series(is_near_lens_candidate, index=candidates.index) & ~is_lens_candidate
    remaining = candidates[~is_lens_candidate & ~is_near_lens_candidate].reset_index(drop=True)
    return remaining, int(is_lens_candidate.sum()), int(is_near_lens_candidate.sum())


def match_in_brightness(lens_magnitudes, candidate_magnitudes, number_to_select, random_generator):
    """Nearest-neighbour matched sampling in magnitude, without replacement.

    The lenses are visited in random order, and the list of lenses is repeated as often as needed.
    At each visit, the not yet selected candidate with the closest magnitude is selected. Because
    every lens is visited equally often, the selected candidates reproduce the brightness
    distribution of the lenses.

    Returns a list of (candidate row, lens row, magnitude difference candidate minus lens)."""
    order_by_magnitude = np.argsort(candidate_magnitudes, kind="stable")   # stable: identical in every run
    sorted_magnitudes = candidate_magnitudes[order_by_magnitude]
    already_selected = np.zeros(len(order_by_magnitude), dtype=bool)
    lens_visit_order = []
    while len(lens_visit_order) < number_to_select:
        lens_visit_order.extend(random_generator.permutation(len(lens_magnitudes)))
    selections = []
    for lens_row in lens_visit_order[:number_to_select]:
        lens_magnitude = lens_magnitudes[lens_row]
        insertion_position = bisect.bisect_left(sorted_magnitudes, lens_magnitude)
        fainter_side, brighter_side = insertion_position - 1, insertion_position
        while fainter_side >= 0 and already_selected[fainter_side]:
            fainter_side -= 1
        while brighter_side < len(order_by_magnitude) and already_selected[brighter_side]:
            brighter_side += 1
        difference_below = (lens_magnitude - sorted_magnitudes[fainter_side]
                            if fainter_side >= 0 else np.inf)
        difference_above = (sorted_magnitudes[brighter_side] - lens_magnitude
                            if brighter_side < len(order_by_magnitude) else np.inf)
        closest = fainter_side if difference_below <= difference_above else brighter_side
        already_selected[closest] = True
        selections.append((order_by_magnitude[closest], lens_row,
                           float(sorted_magnitudes[closest] - lens_magnitude)))
    return selections


def add_benchmark_membership(nonlens_table, number_of_lenses):
    """Add the selection rank and the membership columns: for each benchmark size, the first
    (size minus number of lenses) non-lenses belong to that benchmark; the rest are reserves."""
    nonlens_table.insert(0, "rank", np.arange(1, len(nonlens_table) + 1))
    for size_name, size in BENCHMARK_SIZES.items():
        nonlens_table[f"in_{size_name}"] = nonlens_table["rank"] <= size - number_of_lenses
    nonlens_table["reserve"] = nonlens_table["rank"] > max(BENCHMARK_SIZES.values()) - number_of_lenses
    return nonlens_table


def main():
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    progress_report = ProgressReport()
    random_generator = np.random.default_rng(RANDOM_SEED)
    lens_catalogue = pd.read_csv(LENS_CATALOGUE)
    lens_properties = pd.read_csv(LENS_PROPERTIES)
    lens_properties["magnitude"] = magnitude_from_flux(pd.to_numeric(lens_properties.flux_detection_total,
                                                                     errors="coerce"))
    lens_label = pd.to_numeric(lens_properties.phz_classification, errors="coerce")

    # ---- Lenses: the same 487 in R and S (every lens with a classification label from 1 to 7) ----
    lenses = lens_properties[lens_label.isin(LABELS_R)]
    lenses.to_csv(OUTPUT_FOLDER / "lenses.csv", index=False)
    number_of_lenses = len(lenses)
    progress_report(f"Lenses: {number_of_lenses} in both R and S (labels 1-7); "
                    f"not used {len(lens_properties) - number_of_lenses} (no class, unclassified or missing)")
    nonlenses_needed = max(BENCHMARK_SIZES.values()) - number_of_lenses
    for size_name, size in BENCHMARK_SIZES.items():
        progress_report(f"  {size_name}: R and S each = {number_of_lenses} lenses + "
                        f"{size - number_of_lenses} non-lenses")
    number_to_select = int(np.ceil(nonlenses_needed * (1 + RESERVE_FRACTION)))

    # ---- R: randomly drawn classified objects ----
    progress_report("\nR: reading the pseudo-random subset of classified objects (may take several minutes) ...")
    candidates_R = read_catalogue_subset(
        "SELECT merged_catalogue.object_id, merged_catalogue.right_ascension, merged_catalogue.declination, "
        "merged_catalogue.flux_detection_total, classification.phz_classification AS label "
        "FROM catalogue.mer_catalogue AS merged_catalogue "
        "JOIN catalogue.phz_classification AS classification "
        "ON merged_catalogue.object_id = classification.object_id "
        f"WHERE {SUBSET_CONDITION_R} AND merged_catalogue.spurious_flag = 0 "
        f"AND classification.phz_classification IN {tuple(LABELS_R)}", progress_report)
    progress_report(f"  subset: {len(candidates_R):,} objects; labels "
                    f"{candidates_R.label.value_counts().sort_index().to_dict()}")
    candidates_R, removed_lens_candidates, removed_neighbours = remove_lens_candidates_and_neighbours(
        candidates_R, lens_catalogue)
    progress_report(f"  removed: {removed_lens_candidates} lens candidates, {removed_neighbours} within "
                    f"{EXCLUSION_RADIUS_ARCSEC:.0f} arcsec of one")
    if len(candidates_R) < number_to_select:
        progress_report(f"  ERROR: only {len(candidates_R)} candidates for {number_to_select} needed; "
                        "enlarge the subset.")
        sys.exit(1)
    random_order = random_generator.permutation(len(candidates_R))[:number_to_select]
    nonlenses_R = candidates_R.iloc[random_order].reset_index(drop=True)
    nonlenses_R["magnitude"] = magnitude_from_flux(nonlenses_R.flux_detection_total)
    nonlenses_R = add_benchmark_membership(nonlenses_R, number_of_lenses)
    nonlenses_R.to_csv(OUTPUT_FOLDER / "nonlenses_R.csv", index=False)
    progress_report(f"  selected {len(nonlenses_R):,} (including {number_to_select - nonlenses_needed} reserves); "
                    f"labels {nonlenses_R.label.value_counts().sort_index().to_dict()}")

    # Representativeness of the R subset: galaxies in the subset compared with all Q1 galaxies.
    subset_galaxy_magnitudes = magnitude_from_flux(
        candidates_R.flux_detection_total[candidates_R.label.isin(LABELS_S)
                                          & (candidates_R.flux_detection_total > 0)])
    progress_report("  representativeness: percentage of galaxies brighter than each magnitude "
                    "(subset vs all Q1):")
    for magnitude_limit, percent_all_q1 in PERCENT_OF_Q1_GALAXIES_BRIGHTER.items():
        percent_subset = 100 * (subset_galaxy_magnitudes < magnitude_limit).mean()
        progress_report(f"    {magnitude_limit}: {percent_subset:6.2f}%  vs  {percent_all_q1:6.2f}%")

    # ---- S: galaxies matched to the lenses in brightness ----
    faint_magnitude_limit = lenses.magnitude.max() + FAINT_LIMIT_MARGIN_MAGNITUDES
    faint_flux_limit = 10 ** ((23.9 - faint_magnitude_limit) / 2.5)
    progress_report(f"\nS: reading the pseudo-random subset of galaxies brighter than magnitude "
                    f"{faint_magnitude_limit:.2f} (may take several minutes) ...")
    candidates_S = read_catalogue_subset(
        "SELECT merged_catalogue.object_id, merged_catalogue.right_ascension, merged_catalogue.declination, "
        "merged_catalogue.flux_detection_total, classification.phz_classification AS label "
        "FROM catalogue.mer_catalogue AS merged_catalogue "
        "JOIN catalogue.phz_classification AS classification "
        "ON merged_catalogue.object_id = classification.object_id "
        f"WHERE {SUBSET_CONDITION_S} AND merged_catalogue.spurious_flag = 0 "
        f"AND classification.phz_classification IN {tuple(LABELS_S)} "
        f"AND merged_catalogue.flux_detection_total > {faint_flux_limit:.6f}", progress_report)
    progress_report(f"  subset: {len(candidates_S):,} galaxies")
    candidates_S, removed_lens_candidates, removed_neighbours = remove_lens_candidates_and_neighbours(
        candidates_S, lens_catalogue)
    progress_report(f"  removed: {removed_lens_candidates} lens candidates, {removed_neighbours} within "
                    f"{EXCLUSION_RADIUS_ARCSEC:.0f} arcsec of one")
    candidates_S["magnitude"] = magnitude_from_flux(candidates_S.flux_detection_total)
    matches = match_in_brightness(lenses.magnitude.to_numpy(), candidates_S.magnitude.to_numpy(),
                                  number_to_select, random_generator)
    nonlenses_S = candidates_S.iloc[[candidate_row for candidate_row, _, _ in matches]].reset_index(drop=True)
    nonlenses_S["matched_lens_id"] = [lenses.id_str.iloc[lens_row] for _, lens_row, _ in matches]
    nonlenses_S["magnitude_difference"] = [difference for _, _, difference in matches]
    nonlenses_S = add_benchmark_membership(nonlenses_S, number_of_lenses)
    nonlenses_S.to_csv(OUTPUT_FOLDER / "nonlenses_S.csv", index=False)
    progress_report(f"  selected {len(nonlenses_S):,} (including {number_to_select - nonlenses_needed} reserves)")

    # Quality of the brightness matching
    absolute_differences = nonlenses_S.magnitude_difference.abs()
    progress_report(f"  brightness difference to the matched lens: median {absolute_differences.median():.4f} mag, "
                    f"95% below {absolute_differences.quantile(0.95):.4f}, largest {absolute_differences.max():.4f}")
    progress_report("  brightness distribution (percentiles 5 / 25 / 50 / 75 / 95):")
    for group_name, magnitudes in (("lenses", lenses.magnitude),
                                   ("S non-lenses (10k)", nonlenses_S.magnitude[nonlenses_S.in_10k])):
        progress_report(f"    {group_name:20s} " + " / ".join(
            f"{value:.2f}" for value in np.percentile(magnitudes, [5, 25, 50, 75, 95])))
    bright_end = nonlenses_S.merge(lenses[["id_str", "magnitude"]].rename(columns={"magnitude": "lens_magnitude"}),
                                   left_on="matched_lens_id", right_on="id_str")
    bright_end = bright_end[bright_end.lens_magnitude < 19]
    if len(bright_end):
        progress_report(f"  bright end (lenses brighter than magnitude 19): {bright_end.matched_lens_id.nunique()} "
                        f"lenses, largest difference {bright_end.magnitude_difference.abs().max():.4f} mag")

    (OUTPUT_FOLDER / "selection_report.txt").write_text("\n".join(progress_report.lines) + "\n")
    progress_report(f"\nLists and report saved in {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
