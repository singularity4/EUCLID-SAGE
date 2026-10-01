#!/usr/bin/env python3
r"""
quickstart.py - load EUCLID-SAGE benchmark and evaluate a (trivial) score

Run from the repository folder:
    python examples/quickstart.py

The example uses the central brightness of each stamp as a placeholder "lens score" only to show
the interface; replace it with the scores of a real lens discovery method.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from euclid_sage import evaluate, load_benchmark, load_stamps   # noqa: E402

for benchmark_name in ("EUCLID-SAGE-1k-R_VIS", "EUCLID-SAGE-1k-S_VIS", "EUCLID-SAGE-1k-D_VIS"):
    table = load_benchmark(benchmark_name)
    stamps = load_stamps(table)                                   # (1000, 101, 101), raw values in ADU/s
    print(f"{benchmark_name}: {len(table)} stamps, {int(table.label.sum())} lenses, array {stamps.shape}")

    # Placeholder score: mean value of the central 11 x 11 pixels, scaled to [0, 1] by rank.
    central_brightness = stamps[:, 45:56, 45:56].mean(axis=(1, 2))
    placeholder_score = np.argsort(np.argsort(central_brightness)) / (len(central_brightness) - 1)

    results = evaluate(table, placeholder_score, threshold=0.5)
    print(f"  placeholder score: AUROC {results['AUROC']}, completeness {results['completeness']}, "
          f"false-alarm rate {results['false_alarm_rate']}")
