#!/usr/bin/env python3
r"""
euclid_sage.py - loading and evaluation tools for the EUCLID-SAGE benchmarks 

Part of EUCLID-SAGE: Strong Gravitational Lens Benchmark Suite for Euclid.

Loading
    load_benchmark("EUCLID-SAGE-1k-R_VIS")         -> table (pandas DataFrame), one row per stamp
    load_stamps(table)                             -> array (number of stamps, 101, 101), raw values in ADU/s
    stamp_in_microjansky(pixel_values, zero_point) -> pixel values in microjansky per pixel

Evaluation (method-independent: any classifier that assigns a lens score to each stamp)
    evaluate(table, scores, threshold) -> dictionary of standard metrics
    A predictions file can also be scored from the command line:
        python euclid_sage.py evaluate EUCLID-SAGE-1k-D_VIS my_predictions.csv --threshold 0.5
    where my_predictions.csv has the columns "stamp_file" (as in the benchmark table) and "score"
    (larger = more likely a lens).

Metrics
    AUROC              area under the receiver operating characteristic curve of the scores, computed
                       from ranks (probability that a lens scores higher than a non-lens; ties count half)
    completeness       fraction of lenses with score >= threshold (true-positive rate)
    false_alarm_rate   fraction of non-lenses with score >= threshold (false-positive rate)
    precision          fraction of stamps with score >= threshold that are lenses, in the benchmark
    precision_at_f     precision expected at a lens fraction f other than the benchmark's,
                       C f / (C f + F (1 - f)), with completeness C and false-alarm rate F; reported for
                       the lens fractions in REFERENCE_LENS_FRACTIONS
    completeness_A/B   completeness for lenses of expert grade A and grade B
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_FOLDER = Path(__file__).resolve().parent
BENCHMARK_FOLDER = REPOSITORY_FOLDER / "benchmarks"
# Approximate lens fractions of the underlying Q1 populations (see docs/DESIGN.md, Sect. 8):
# about 2 x 10^-4 among galaxies as bright as the lenses (S and D) and 2 x 10^-5 among all classified objects (R).
REFERENCE_LENS_FRACTIONS = {"bright_galaxies_2e-4": 2e-4, "all_classified_objects_2e-5": 2e-5}


def load_benchmark(name, benchmark_folder=BENCHMARK_FOLDER):
    """Benchmark table: one row per stamp, with stamp_file, label (1 lens, 0 non-lens) and metadata."""
    return pd.read_csv(Path(benchmark_folder) / f"{name}.csv")


def load_stamps(table, benchmark_folder=BENCHMARK_FOLDER):
    """All stamps of a table as one array of raw pixel values (ADU/s), in the order of the table."""
    from astropy.io import fits
    return np.stack([fits.getdata(Path(benchmark_folder) / stamp_file).astype(np.float32)
                     for stamp_file in table.stamp_file])


def stamp_in_microjansky(pixel_values, zero_point=24.6):
    """Convert raw pixel values to microjansky per pixel with the stamp's zero point (header MAGZERO)."""
    return pixel_values * 10 ** ((23.9 - zero_point) / 2.5)


def rank_auroc(scores, labels):
    """AUROC from ranks (Mann-Whitney statistic); ties count half."""
    from scipy.stats import rankdata
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    number_of_lenses, number_of_nonlenses = int((labels == 1).sum()), int((labels == 0).sum())
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - number_of_lenses * (number_of_lenses + 1) / 2)
                 / (number_of_lenses * number_of_nonlenses))


def evaluate(table, scores, threshold=0.5):
    """Standard metrics for lens scores given in the order of the table."""
    scores = np.asarray(scores, float)
    if len(scores) != len(table) or np.isnan(scores).any():
        raise ValueError("one finite score per stamp of the benchmark is required")
    labels = table.label.to_numpy().astype(int)
    flagged = scores >= threshold
    lenses, nonlenses = labels == 1, labels == 0
    completeness = float(flagged[lenses].mean())
    false_alarm_rate = float(flagged[nonlenses].mean())
    results = {"stamps": len(table), "lenses": int(lenses.sum()), "nonlenses": int(nonlenses.sum()),
               "threshold": threshold, "AUROC": round(rank_auroc(scores, labels), 4),
               "lenses_found": int(flagged[lenses].sum()), "completeness": round(completeness, 4),
               "false_alarms": int(flagged[nonlenses].sum()), "false_alarm_rate": round(false_alarm_rate, 5),
               "precision": round(float(labels[flagged].mean()), 4) if flagged.any() else None}
    for fraction_name, lens_fraction in REFERENCE_LENS_FRACTIONS.items():
        denominator = completeness * lens_fraction + false_alarm_rate * (1 - lens_fraction)
        results[f"precision_at_{fraction_name}"] = (round(completeness * lens_fraction / denominator, 5)
                                                    if denominator else None)
    for grade in ("A", "B"):
        graded = lenses & (table.grade.astype(str) == grade).to_numpy()
        results[f"completeness_grade_{grade}"] = round(float(flagged[graded].mean()), 4) if graded.any() else None
    return results


def main():
    parser = argparse.ArgumentParser(description="EUCLID-SAGE utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate", help="score a predictions file")
    evaluation.add_argument("benchmark", help="e.g. EUCLID-SAGE-1k-D_VIS")
    evaluation.add_argument("predictions", help="CSV with columns stamp_file and score")
    evaluation.add_argument("--threshold", type=float, default=0.5)
    settings = parser.parse_args()

    table = load_benchmark(settings.benchmark)
    predictions = pd.read_csv(settings.predictions)[["stamp_file", "score"]]
    merged = table.merge(predictions, on="stamp_file", how="left")
    if merged.score.isna().any():
        raise SystemExit(f"{int(merged.score.isna().sum())} stamps of the benchmark have no score")
    for key, value in evaluate(merged, merged.score, settings.threshold).items():
        print(f"{key:40s} {value}")


if __name__ == "__main__":
    main()
