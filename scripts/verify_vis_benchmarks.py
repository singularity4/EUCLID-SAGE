#!/usr/bin/env python3
r"""
verify_vis_benchmarks.py - verification of the black-and-white VIS benchmarks (R, S, and D)

Part of EUCLID-SAGE: Strong Gravitational Lens Benchmark Suite for Euclid.

Checks
    1. Size: 1,000 or 10,000 stamps per benchmark, with 487 lenses in each.
    2. Uniqueness: no object appears twice within a benchmark.
    3. Stamp files: every file exists, has 101 x 101 pixels without missing values, and carries the
       expected header entries (BUNIT = ADU/s, a zero point MAGZERO, OBJTYPE matching the label).
    4. Consistency: the same 487 lenses in all benchmarks; the 1k non-lenses are contained in the
       10k non-lenses.
    5. Non-lenses: none is a lens candidate of any grade in the lens catalogue.
    6. S and D benchmarks: the brightness distribution of the non-lenses matches that of the lenses;
       for D also the Sersic index, Sersic radius and ellipticity (values from the selection lists).
    7. R benchmarks: distribution of the Euclid classification labels among the non-lenses.

Output
    printed results, saved as benchmarks_vis/verification_report.txt

Run from the data folder, after build_vis_benchmarks.py:
    python3 verify_vis_benchmarks.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

# Working folder holding the input catalogues and the outputs: the folder given in the environment
# variable EUCLID_SAGE_DATA_DIR, otherwise the folder the script is run from.
EUCLID_FOLDER = Path(os.environ.get("EUCLID_SAGE_DATA_DIR", ".")).resolve()
BENCHMARK_FOLDER = EUCLID_FOLDER / "benchmarks_vis"
LENS_CATALOGUE = EUCLID_FOLDER / "q1_discovery_engine_lens_catalog.csv"
SELECTION_FOLDER = EUCLID_FOLDER / "nonlens_selection"
BENCHMARK_TYPES = [benchmark_type for benchmark_type in ("R", "S", "D")
                   if (BENCHMARK_FOLDER / f"EUCLID-SAGE-1k-{benchmark_type}_VIS.csv").exists()]
EXPECTED_SIZE_BY_BENCHMARK = {f"EUCLID-SAGE-{size_name}-{benchmark_type}_VIS": size
                              for size_name, size in (("1k", 1000), ("10k", 10000))
                              for benchmark_type in BENCHMARK_TYPES}
EXPECTED_NUMBER_OF_LENSES = 487
STAMP_SIZE_PIXELS = 101

report_lines = []


def report(text=""):
    """Print a line and keep it for verification_report.txt."""
    print(text, flush=True)
    report_lines.append(text)


def report_check(passed, description):
    """Report one check as OK or PROBLEM; returns whether it passed."""
    report(f"  [{'OK' if passed else 'PROBLEM'}] {description}")
    return passed


def problems_with_stamp(stamp_path, expected_object_type):
    """List the problems found in one stamp file (an empty list if there are none)."""
    if not stamp_path.exists():
        return ["file missing"]
    with fits.open(stamp_path) as stamp_file:
        pixel_values, header = stamp_file[0].data, stamp_file[0].header
        problems = []
        if pixel_values is None or pixel_values.shape != (STAMP_SIZE_PIXELS, STAMP_SIZE_PIXELS):
            problems.append(f"shape {None if pixel_values is None else pixel_values.shape}")
        elif not np.all(np.isfinite(pixel_values)):
            problems.append("missing or non-finite pixel values")
        if header.get("BUNIT") != "ADU/s":
            problems.append(f"BUNIT {header.get('BUNIT')}")
        if header.get("MAGZERO") is None:
            problems.append("no MAGZERO")
        if header.get("OBJTYPE") != expected_object_type:
            problems.append(f"OBJTYPE {header.get('OBJTYPE')}")
        return problems


def object_identifier_from_id_str(id_str):
    """Euclid object identifier from a lens catalogue id_str ('<tile>_NEG<digits>' -> -<digits>)."""
    try:
        return int(str(id_str).split("_", 1)[1].replace("NEG", "-"))
    except (IndexError, ValueError):
        return None


def main():
    benchmark_tables = {benchmark_name: pd.read_csv(BENCHMARK_FOLDER / f"{benchmark_name}.csv")
                        for benchmark_name in EXPECTED_SIZE_BY_BENCHMARK}
    all_checks_passed = True

    report("1-3. Size, uniqueness and stamp files")
    problems_by_stamp_file = {}
    for benchmark_name, expected_size in EXPECTED_SIZE_BY_BENCHMARK.items():
        benchmark_table = benchmark_tables[benchmark_name]
        report(f" {benchmark_name}")
        number_of_lenses = int(benchmark_table.label.sum())
        all_checks_passed &= report_check(
            len(benchmark_table) == expected_size and number_of_lenses == EXPECTED_NUMBER_OF_LENSES,
            f"{len(benchmark_table):,} stamps, {number_of_lenses} lenses, "
            f"{int((benchmark_table.label == 0).sum()):,} non-lenses")
        all_checks_passed &= report_check(not benchmark_table.stamp_file.duplicated().any(),
                                          f"duplicates: {int(benchmark_table.stamp_file.duplicated().sum())}")
        stamps_with_problems = {}
        for benchmark_entry in benchmark_table.itertuples():
            if benchmark_entry.stamp_file not in problems_by_stamp_file:
                problems_by_stamp_file[benchmark_entry.stamp_file] = problems_with_stamp(
                    BENCHMARK_FOLDER / benchmark_entry.stamp_file,
                    "lens" if benchmark_entry.label == 1 else "nonlens")
            if problems_by_stamp_file[benchmark_entry.stamp_file]:
                stamps_with_problems[benchmark_entry.stamp_file] = problems_by_stamp_file[benchmark_entry.stamp_file]
        all_checks_passed &= report_check(
            not stamps_with_problems, f"stamp files with problems: {len(stamps_with_problems)}"
            + (f", for example {list(stamps_with_problems.items())[:3]}" if stamps_with_problems else ""))
    zero_points_found = set()
    for stamp_file_name in list(problems_by_stamp_file)[:2000]:
        with fits.open(BENCHMARK_FOLDER / stamp_file_name) as stamp_file:
            zero_points_found.add(stamp_file[0].header.get("MAGZERO"))
    report(f"  zero points found (first 2,000 stamps): {sorted(zero_points_found)}")

    report("\n4. Lenses and nesting")
    lenses_by_benchmark = {benchmark_name: set(benchmark_table.object_id[benchmark_table.label == 1])
                           for benchmark_name, benchmark_table in benchmark_tables.items()}
    distinct_lens_sets = {frozenset(lens_set) for lens_set in lenses_by_benchmark.values()}
    all_checks_passed &= report_check(len(distinct_lens_sets) == 1,
                                      f"the same lenses in all {len(benchmark_tables)} benchmarks")
    for benchmark_type in BENCHMARK_TYPES:
        small_table = benchmark_tables[f"EUCLID-SAGE-1k-{benchmark_type}_VIS"]
        large_table = benchmark_tables[f"EUCLID-SAGE-10k-{benchmark_type}_VIS"]
        small_nonlenses = set(small_table.object_id[small_table.label == 0])
        large_nonlenses = set(large_table.object_id[large_table.label == 0])
        all_checks_passed &= report_check(small_nonlenses <= large_nonlenses,
                                          f"{benchmark_type}: 1k non-lenses are part of the 10k non-lenses")

    report("\n5. Non-lenses are not lens candidates")
    lens_candidate_identifiers = {identifier for identifier in
                                  map(object_identifier_from_id_str, pd.read_csv(LENS_CATALOGUE).id_str.dropna())
                                  if identifier is not None}
    for benchmark_name, benchmark_table in benchmark_tables.items():
        overlap = set(benchmark_table.object_id[benchmark_table.label == 0]) & lens_candidate_identifiers
        all_checks_passed &= report_check(not overlap,
                                          f"{benchmark_name}: non-lenses that are lens candidates: {len(overlap)}")

    report("\n6. S and D: brightness of non-lenses and lenses (VIS magnitude percentiles 5 / 25 / 50 / 75 / 95)")
    for benchmark_name in [name for name in benchmark_tables if "-S_" in name or "-D_" in name]:
        benchmark_table = benchmark_tables[benchmark_name]
        for label_value, group_name in ((1, "lenses"), (0, "non-lenses")):
            magnitudes = benchmark_table.vis_magnitude[benchmark_table.label == label_value].dropna()
            report(f"  {benchmark_name} {group_name:10s} " + " / ".join(
                f"{value:.2f}" for value in np.percentile(magnitudes, [5, 25, 50, 75, 95])))

    if "D" in BENCHMARK_TYPES:
        report("\n6b. D: Sersic index, Sersic radius and ellipticity of non-lenses and lenses "
               "(percentiles 5 / 25 / 50 / 75 / 95)")
        lens_values = pd.read_csv(SELECTION_FOLDER / "lenses.csv")
        nonlens_values = pd.read_csv(SELECTION_FOLDER / "nonlenses_D.csv")
        for benchmark_name in ("EUCLID-SAGE-1k-D_VIS", "EUCLID-SAGE-10k-D_VIS"):
            benchmark_table = benchmark_tables[benchmark_name]
            benchmark_nonlenses = nonlens_values[nonlens_values.object_id.isin(
                benchmark_table.object_id[benchmark_table.label == 0])]
            for column, property_name in (("sersic_sersic_vis_index", "Sersic index"),
                                          ("sersic_sersic_vis_radius", "Sersic radius"),
                                          ("ellipticity", "ellipticity")):
                for group_name, values in (("lenses", lens_values[column]), ("non-lenses", benchmark_nonlenses[column])):
                    values = pd.to_numeric(values, errors="coerce").dropna()
                    report(f"  {benchmark_name} {property_name:13s} {group_name:10s} " + " / ".join(
                        f"{value:.2f}" for value in np.percentile(values, [5, 25, 50, 75, 95])))

    report("\n7. R: Euclid classification labels of the non-lenses "
           "(1 star, 2 galaxy, 3 galaxy and star, 4 quasar, 5-7 combinations)")
    for benchmark_name in ("EUCLID-SAGE-1k-R_VIS", "EUCLID-SAGE-10k-R_VIS"):
        benchmark_table = benchmark_tables[benchmark_name]
        label_counts = benchmark_table.phz_label[benchmark_table.label == 0].value_counts().sort_index()
        report(f"  {benchmark_name}: {label_counts.to_dict()}")

    report(f"\nOverall: {'all checks passed' if all_checks_passed else 'PROBLEMS FOUND - see above'}")
    (BENCHMARK_FOLDER / "verification_report.txt").write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {BENCHMARK_FOLDER / 'verification_report.txt'}")


if __name__ == "__main__":
    main()
