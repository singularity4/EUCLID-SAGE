#!/usr/bin/env python3
r"""
lens_catalogue_properties.py - Euclid classification labels and catalogue properties of the lenses

Part of EUCLID-SAGE: Strong Gravitational Lens Benchmark Suite for Euclid.

Purpose
    Collect, for every grade A and B lens candidate of the main search, the official Euclid Q1
    catalogue values used to build the benchmarks, and save them as lens_catalogue_properties.csv
    (required by select_nonlenses.py). The script also prints four diagnostics that informed the
    benchmark design:
    1. the Euclid classification label of the lenses (phz_classification);
    2. the brightness, size, shape and morphology of the lens galaxies;
    3. which of two flux columns holds the total flux (their descriptions in the archive are
       shifted by one line; the total flux is the column with much larger values than its error);
    4. the number of Q1 objects per classification label.

The classification label
    phz_classification is a sum of flags: +1 star, +2 galaxy, +4 quasar (Euclid Q1 Data Product
    Description Document). A value of 2 therefore means "accepted only as a galaxy". The column
    description in the archive table states the opposite order (+1 galaxy, +2 star); the Data Product
    Description and the data themselves (25.1 million Q1 objects with value 2, 0.7 million with
    value 1) confirm the order used here.

Output
    lens_catalogue_properties.csv  one row per lens, with every catalogue value listed below
    printed diagnostics

Run from the data folder:
    python3 lens_catalogue_properties.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.esa.euclid import Euclid

# Working folder holding the input catalogues and the outputs: the folder given in the environment
# variable EUCLID_SAGE_DATA_DIR, otherwise the folder the script is run from.
EUCLID_FOLDER = Path(os.environ.get("EUCLID_SAGE_DATA_DIR", ".")).resolve()
LENS_CATALOGUE = EUCLID_FOLDER / "q1_discovery_engine_lens_catalog.csv"
OUTPUT_FILE = EUCLID_FOLDER / "lens_catalogue_properties.csv"

# Object identifiers per catalogue request; short requests are answered quickly by the archive.
OBJECTS_PER_CATALOGUE_REQUEST = 100

# Catalogue values collected for every lens, by archive table.
COLUMNS_BY_CATALOGUE_TABLE = {
    "catalogue.phz_classification": "object_id, phz_classification, phz_gal_prob, phz_star_prob, phz_qso_prob",
    "catalogue.mer_catalogue": ("object_id, flux_detection_total, fluxerr_segmentation, flux_segmentation, "
                                "flux_vis_sersic, segmentation_area, kron_radius, ellipticity, "
                                "point_like_prob, spurious_prob, det_quality_flag, vis_det"),
    "catalogue.mer_morphology": ("object_id, smooth_or_featured_smooth, smooth_or_featured_featured_or_disk, "
                                 "smooth_or_featured_artifact_star_zoom, sersic_sersic_vis_index, "
                                 "sersic_sersic_vis_radius, etg_or_ltg"),
}
ONLY_GALAXY_LABEL = 2   # phz_classification value for "accepted only as a galaxy"


def read_catalogue_values(table_name, column_list, object_identifiers):
    """Read the listed columns of one archive table for the given Euclid object identifiers."""
    partial_tables = []
    for first_index in range(0, len(object_identifiers), OBJECTS_PER_CATALOGUE_REQUEST):
        identifier_text = ", ".join(str(int(identifier)) for identifier in
                                    object_identifiers[first_index:first_index + OBJECTS_PER_CATALOGUE_REQUEST])
        catalogue_request = f"SELECT {column_list} FROM {table_name} WHERE object_id IN ({identifier_text})"
        partial_tables.append(Euclid.launch_job(catalogue_request).get_results().to_pandas())
    return pd.concat(partial_tables, ignore_index=True) if partial_tables else pd.DataFrame()


def describe_distribution(values):
    """Median and 5th/95th percentiles of the valid (numeric) values, as text."""
    valid_values = pd.to_numeric(values, errors="coerce").dropna()
    if valid_values.empty:
        return "no values"
    percentile_5, median, percentile_95 = np.percentile(valid_values, [5, 50, 95])
    return (f"median {median:.4g}  (5%: {percentile_5:.4g}, 95%: {percentile_95:.4g}),  "
            f"{len(valid_values)} values")


def main():
    lens_catalogue = pd.read_csv(LENS_CATALOGUE)
    lenses = lens_catalogue[(lens_catalogue.subset == "discovery_engine")
                            & lens_catalogue.grade.isin(["A", "B"])][["id_str", "grade"]].copy()
    # The Euclid object identifier is taken from id_str ("<tile>_<object identifier>", with the minus
    # sign written as "NEG"), which stores it exactly as text. The catalogue's own object_id column is
    # not used: it has empty rows (candidates from other searches), so it is read as floating-point
    # numbers, which cannot represent 18-digit identifiers exactly.
    lenses["object_id"] = [int(identifier_text.split("_", 1)[1].replace("NEG", "-"))
                           for identifier_text in lenses.id_str]
    object_identifiers = lenses.object_id.tolist()
    print(f"Lenses: {len(lenses)}")

    lens_properties = lenses.copy()
    for table_name, column_list in COLUMNS_BY_CATALOGUE_TABLE.items():
        table_values = read_catalogue_values(table_name, column_list, object_identifiers)
        print(f"  {table_name}: values found for {table_values.object_id.nunique()} of {len(lenses)} lenses")
        lens_properties = lens_properties.merge(table_values.drop_duplicates("object_id"),
                                                on="object_id", how="left")
    lens_properties.to_csv(OUTPUT_FILE, index=False)

    # 1. Classification label of the lenses
    print(f"\n1. Euclid classification label of the lenses (phz_classification; "
          f"{ONLY_GALAXY_LABEL} = only galaxy):")
    print("   " + str(lens_properties.phz_classification.value_counts(dropna=False).to_dict()))
    number_only_galaxy = int((lens_properties.phz_classification == ONLY_GALAXY_LABEL).sum())
    print(f"   Lenses labelled only galaxy: {number_only_galaxy} of {len(lens_properties)}")

    # 2. Properties of the lens galaxies
    print("\n2. Lens galaxies in the official catalogue:")
    for column in ["flux_detection_total", "flux_segmentation", "flux_vis_sersic", "segmentation_area",
                   "kron_radius", "ellipticity", "point_like_prob", "spurious_prob", "sersic_sersic_vis_index",
                   "sersic_sersic_vis_radius"]:
        print(f"   {column:28s} {describe_distribution(lens_properties[column])}")
    # The morphology columns are parameters of a Dirichlet distribution; the expected fraction of
    # "smooth" answers is smooth / (smooth + featured + artefact).
    dirichlet_total = (lens_properties.smooth_or_featured_smooth
                       + lens_properties.smooth_or_featured_featured_or_disk
                       + lens_properties.smooth_or_featured_artifact_star_zoom)
    lens_properties["smooth_fraction"] = lens_properties.smooth_or_featured_smooth / dirichlet_total
    print(f"   {'smooth_fraction (derived)':28s} {describe_distribution(lens_properties.smooth_fraction)}")
    print(f"   etg_or_ltg values: {lens_properties.etg_or_ltg.value_counts(dropna=False).to_dict()}")

    # 3. Which column holds the total flux?
    print("\n3. Typical values of the two columns with shifted descriptions (uJy):")
    print(f"   flux_detection_total   {describe_distribution(lens_properties.flux_detection_total)}")
    print(f"   fluxerr_segmentation   {describe_distribution(lens_properties.fluxerr_segmentation)}")
    print("   The total flux is the column with the much larger values.")

    # 4. Q1 objects per classification label
    print("\n4. Counting Q1 objects per classification label (may take a minute) ...")
    try:
        label_count_request = Euclid.launch_job_async(
            "SELECT phz_classification, COUNT(*) AS number_of_objects FROM catalogue.phz_classification "
            "GROUP BY phz_classification")
        label_counts = label_count_request.get_results().to_pandas().sort_values("phz_classification")
        print("   Objects per label: " + ", ".join(
            f"{int(label_row.phz_classification) if pd.notna(label_row.phz_classification) else 'none'}: "
            f"{int(label_row.number_of_objects):,}" for label_row in label_counts.itertuples()))
    except Exception as error:
        print(f"   (the label count could not be obtained: {error})")

    print(f"\nLens catalogue values saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
