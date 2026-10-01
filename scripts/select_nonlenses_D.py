#!/usr/bin/env python3
r"""
select_nonlenses_D.py - Sample non-lenses of the D (difficult) benchmarks: galaxies matched to the lenses in
brightness, light concentration, size and roundness.

Part of EUCLID-SAGE: Strong Gravitational Lens Benchmark Suite for Euclid.

Purpose
    The S benchmarks match non-lenses to the strong lenses in brightness only. Lens galaxies, however, are
    mostly massive early-type (elliptical) galaxies, so a classifier could still separate S partly by
    galaxy type. The D benchmarks reduce this possibility: each non-lens is a galaxy that resembles a
    lens galaxy in brightness, in the concentration of its light, in its size and in its roundness, so
    that the lensing features (arcs, rings) are the main remaining difference.

    The four properties are those that separated lenses from S non-lenses in the separability analysis of docs/DESIGN.md, 
    Sect. 8 (AUROC of single properties in S: Sersic index 0.66-0.69, ellipticity 0.38-0.41, Sersic radius 0.59-0.62; 
    brightness 0.50 by construction).

Selection rules (the same as for S, plus three matched properties)

    Lenses: the same 487 as in R and S (nonlens_selection/lenses.csv).
    D non-lenses: galaxies (classification labels 2 and 3), not flagged as a false detection
    (spurious_flag = 0), not a lens candidate of any grade or subset, at least 12 arcsec from every
    lens candidate, with valid Sersic values; random seed 42; the 1k non-lenses are the first
    entries of the 10k list; 10% reserves.

    Matched properties (official Euclid Q1 catalogues):
      - VIS magnitude from flux_detection_total (catalogue.mer_catalogue): brightness;
      - Sersic index, sersic_sersic_vis_index (catalogue.mer_morphology): concentration of the light,
        about 4 for elliptical galaxies and about 1 for disks;
      - Sersic radius, sersic_sersic_vis_radius (catalogue.mer_morphology): size;
      - ellipticity (catalogue.mer_catalogue): roundness (0 for a round image).

    The index and the radius are used as logarithms, because both are positive and span a wide range
    of values. Each of the four properties is divided by its standard deviation among the candidates,
    so that all four carry equal weight.

    Note: the catalogue values of a lens system include the light of its arcs (a ring makes a system
    appear rounder and larger), so each non-lens is matched to the whole lens system, not to the lens
    galaxy alone. This makes D harder rather than easier.

Matching (the method of S, extended to four properties)
    Nearest-neighbour matched sampling without replacement: the lenses are visited in random order
    (the list is repeated as often as needed); at each visit the not yet selected candidate closest to
    the lens in the four standardised properties is selected. Every lens is visited equally often,
    so the selected non-lenses reproduce the joint distribution of the four properties among the lenses.

Sampling from the archive
    As for S: a fixed pseudo-random subset (last two digits of the object identifier below 10, about
    10% of all objects) of galaxies brighter than the faintest lens plus 0.2 magnitudes, with the
    Sersic values read from catalogue.mer_morphology and the ellipticity from catalogue.mer_catalogue; 
    the rows are sorted by object identifier so that the seeded selection is reproducible. 
    Objects may also appear in the R or S benchmarks.

Output (Euclid/nonlens_selection/)
    nonlenses_D.csv        ordered non-lenses: rank, object_id, position, label, flux, magnitude,
                           Sersic index and radius, ellipticity, matched_lens_id, standardised distance to the
                           matched lens, in_1k, in_10k, reserve
    selection_report_D.txt the printed diagnostics

Run from the data folder, after select_nonlenses.py:
    python3 select_nonlenses_D.py
Then download the stamps and build the D tables:
    python3 build_vis_benchmarks.py --types D
"""
import os
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
SELECTION_FOLDER = EUCLID_FOLDER / "nonlens_selection"

RANDOM_SEED = 42
BENCHMARK_SIZES = {"1k": 1000, "10k": 10000}
RESERVE_FRACTION = 0.10
EXCLUSION_RADIUS_ARCSEC = 12.0
LABELS_D = (2, 3)                                                      # galaxy; galaxy and star
SUBSET_CONDITION = "MOD(ABS(merged_catalogue.object_id), 100) < 10"    # about 10% of all objects
FAINT_LIMIT_MARGIN_MAGNITUDES = 0.2
MATCHED_PROPERTIES = ["magnitude", "log10_sersic_index", "log10_sersic_radius", "ellipticity"]


class ProgressReport:
    """Prints each line and keeps a copy for selection_report_D.txt."""
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)


def magnitude_from_flux(flux_in_microjansky):
    """AB magnitude from the total VIS flux in microjansky (column flux_detection_total)."""
    return -2.5 * np.log10(flux_in_microjansky) + 23.9


def object_identifier_from_id_str(id_str):
    """Euclid object identifier from a lens catalogue id_str ('<tile>_NEG<digits>' -> -<digits>)."""
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


def remove_lens_candidates_and_neighbours(candidates, lens_catalogue):
    """Remove every lens candidate (any grade, any subset) and every object within the exclusion
    radius of one (as in select_nonlenses.py)."""
    lens_candidate_identifiers = {identifier for identifier in
                                  map(object_identifier_from_id_str, lens_catalogue.id_str.dropna())
                                  if identifier is not None}
    is_lens_candidate = candidates.object_id.isin(lens_candidate_identifiers)
    candidates_with_position = lens_catalogue.dropna(subset=["right_ascension", "declination"])
    sky_search_tree = cKDTree(unit_vectors(candidates.right_ascension.values, candidates.declination.values))
    exclusion_chord_length = 2 * np.sin(np.radians(EXCLUSION_RADIUS_ARCSEC / 3600) / 2)
    is_near_lens_candidate = np.zeros(len(candidates), dtype=bool)
    for nearby_rows in sky_search_tree.query_ball_point(
            unit_vectors(candidates_with_position.right_ascension.values,
                         candidates_with_position.declination.values), exclusion_chord_length):
        is_near_lens_candidate[nearby_rows] = True
    is_near_lens_candidate = pd.Series(is_near_lens_candidate, index=candidates.index) & ~is_lens_candidate
    remaining = candidates[~is_lens_candidate & ~is_near_lens_candidate].reset_index(drop=True)
    return remaining, int(is_lens_candidate.sum()), int(is_near_lens_candidate.sum())


def add_matched_properties(table, index_column, radius_column, flux_column):
    """Magnitude, logarithms of Sersic index and radius, and ellipticity (invalid values become missing)."""
    table = table.copy()
    flux = pd.to_numeric(table[flux_column], errors="coerce")
    sersic_index = pd.to_numeric(table[index_column], errors="coerce")
    sersic_radius = pd.to_numeric(table[radius_column], errors="coerce")
    table["magnitude"] = magnitude_from_flux(flux.where(flux > 0))
    table["log10_sersic_index"] = np.log10(sersic_index.where(sersic_index > 0))
    table["log10_sersic_radius"] = np.log10(sersic_radius.where(sersic_radius > 0))
    ellipticity = pd.to_numeric(table["ellipticity"], errors="coerce")
    table["ellipticity"] = ellipticity.where((ellipticity >= 0) & (ellipticity <= 1))
    return table


def match_in_four_properties(lens_properties, candidate_properties, number_to_select, random_generator):
    """Nearest-neighbour matched sampling without replacement in the standardised properties.
    Returns a list of (candidate row, lens row, standardised distance)."""
    property_spread = candidate_properties.std(axis=0)
    candidate_points = candidate_properties / property_spread
    lens_points = lens_properties / property_spread
    property_search_tree = cKDTree(candidate_points)
    already_selected = np.zeros(len(candidate_points), dtype=bool)
    lens_visit_order = []
    while len(lens_visit_order) < number_to_select:
        lens_visit_order.extend(random_generator.permutation(len(lens_points)))
    selections = []
    for lens_row in lens_visit_order[:number_to_select]:
        neighbours_to_check = 16
        while True:
            distances, nearest_rows = property_search_tree.query(
                lens_points[lens_row], k=min(neighbours_to_check, len(candidate_points)))
            distances, nearest_rows = np.atleast_1d(distances), np.atleast_1d(nearest_rows)
            unused = ~already_selected[nearest_rows]
            if unused.any():
                first_unused = int(np.argmax(unused))
                chosen_row = int(nearest_rows[first_unused])
                already_selected[chosen_row] = True
                selections.append((chosen_row, lens_row, float(distances[first_unused])))
                break
            if neighbours_to_check >= len(candidate_points):
                raise RuntimeError("no unused candidates left")
            neighbours_to_check *= 4
    return selections


def main():
    progress_report = ProgressReport()
    random_generator = np.random.default_rng(RANDOM_SEED)
    lens_catalogue = pd.read_csv(LENS_CATALOGUE)
    lenses = add_matched_properties(pd.read_csv(SELECTION_FOLDER / "lenses.csv"),
                                    "sersic_sersic_vis_index", "sersic_sersic_vis_radius", "flux_detection_total")
    number_of_lenses = len(lenses)
    lenses_with_values = lenses.dropna(subset=MATCHED_PROPERTIES)
    progress_report(f"Lenses: {number_of_lenses}; with all four matched properties: {len(lenses_with_values)}")
    nonlenses_needed = max(BENCHMARK_SIZES.values()) - number_of_lenses
    number_to_select = int(np.ceil(nonlenses_needed * (1 + RESERVE_FRACTION)))

    faint_magnitude_limit = lenses.magnitude.max() + FAINT_LIMIT_MARGIN_MAGNITUDES
    faint_flux_limit = 10 ** ((23.9 - faint_magnitude_limit) / 2.5)
    progress_report(f"D: reading the pseudo-random subset of galaxies brighter than magnitude "
                    f"{faint_magnitude_limit:.2f}, with Sersic values and ellipticity (may take several minutes) ...")
    archive_request = Euclid.launch_job_async(
        "SELECT merged_catalogue.object_id, merged_catalogue.right_ascension, merged_catalogue.declination, "
        "merged_catalogue.flux_detection_total, merged_catalogue.ellipticity, "
        "classification.phz_classification AS label, "
        "morphology.sersic_sersic_vis_index, morphology.sersic_sersic_vis_radius "
        "FROM catalogue.mer_catalogue AS merged_catalogue "
        "JOIN catalogue.phz_classification AS classification ON merged_catalogue.object_id = classification.object_id "
        "JOIN catalogue.mer_morphology AS morphology ON merged_catalogue.object_id = morphology.object_id "
        f"WHERE {SUBSET_CONDITION} AND merged_catalogue.spurious_flag = 0 "
        f"AND classification.phz_classification IN {tuple(LABELS_D)} "
        f"AND merged_catalogue.flux_detection_total > {faint_flux_limit:.6f}")
    if archive_request is None:
        progress_report("  ERROR: the archive request failed; nothing was selected.")
        sys.exit(1)
    candidates = archive_request.get_results().to_pandas().sort_values("object_id").reset_index(drop=True)
    progress_report(f"  subset: {len(candidates):,} galaxies")
    candidates, removed_lens_candidates, removed_neighbours = remove_lens_candidates_and_neighbours(
        candidates, lens_catalogue)
    progress_report(f"  removed: {removed_lens_candidates} lens candidates, {removed_neighbours} within "
                    f"{EXCLUSION_RADIUS_ARCSEC:.0f} arcsec of one")
    candidates = add_matched_properties(candidates, "sersic_sersic_vis_index", "sersic_sersic_vis_radius",
                                        "flux_detection_total")
    without_values = int(candidates[MATCHED_PROPERTIES].isna().any(axis=1).sum())
    candidates = candidates.dropna(subset=MATCHED_PROPERTIES).reset_index(drop=True)
    progress_report(f"  without valid values of the four properties (left out): {without_values:,}; "
                    f"candidates: {len(candidates):,}")

    matches = match_in_four_properties(lenses_with_values[MATCHED_PROPERTIES].to_numpy(float),
                                        candidates[MATCHED_PROPERTIES].to_numpy(float),
                                        number_to_select, random_generator)
    nonlenses_D = candidates.iloc[[candidate_row for candidate_row, _, _ in matches]].reset_index(drop=True)
    nonlenses_D["matched_lens_id"] = [lenses_with_values.id_str.iloc[lens_row] for _, lens_row, _ in matches]
    nonlenses_D["standardised_distance"] = np.round([distance for _, _, distance in matches], 5)
    nonlenses_D.insert(0, "rank", np.arange(1, len(nonlenses_D) + 1))
    for size_name, size in BENCHMARK_SIZES.items():
        nonlenses_D[f"in_{size_name}"] = nonlenses_D["rank"] <= size - number_of_lenses
    nonlenses_D["reserve"] = nonlenses_D["rank"] > nonlenses_needed
    nonlenses_D.to_csv(SELECTION_FOLDER / "nonlenses_D.csv", index=False)
    progress_report(f"  selected {len(nonlenses_D):,} (including {number_to_select - nonlenses_needed} reserves); "
                    f"labels {nonlenses_D.label.value_counts().sort_index().to_dict()}")

    # Quality of the matching
    distances = nonlenses_D.standardised_distance
    progress_report(f"  standardised distance to the matched lens: median {distances.median():.4f}, "
                    f"95% below {distances.quantile(0.95):.4f}, largest {distances.max():.4f}")
    progress_report("  distributions (percentiles 5 / 25 / 50 / 75 / 95), lenses vs D non-lenses (10k):")
    in_10k = nonlenses_D[nonlenses_D.in_10k]
    for property_name, lens_values, nonlens_values in (
            ("VIS magnitude", lenses_with_values.magnitude, in_10k.magnitude),
            ("Sersic index", 10 ** lenses_with_values.log10_sersic_index, 10 ** in_10k.log10_sersic_index),
            ("Sersic radius", 10 ** lenses_with_values.log10_sersic_radius, 10 ** in_10k.log10_sersic_radius),
            ("ellipticity", lenses_with_values.ellipticity, in_10k.ellipticity)):
        for group_name, values in (("lenses", lens_values), ("D non-lenses", nonlens_values)):
            progress_report(f"    {property_name:14s} {group_name:13s} " + " / ".join(
                f"{value:.2f}" for value in np.percentile(values, [5, 25, 50, 75, 95])))

    (SELECTION_FOLDER / "selection_report_D.txt").write_text("\n".join(progress_report.lines) + "\n")
    progress_report(f"\nList and report saved in {SELECTION_FOLDER}")


if __name__ == "__main__":
    main()
