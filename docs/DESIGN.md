# EUCLID-SAGE: Benchmark design

This document describes design and construction of the black-and-white (VIS) benchmarks. Script names refer to `scripts/`.

## 1. Overview

1. Lens sample: grade A and B candidates of the Euclid Q1 Strong Lensing Discovery Engine with a Euclid
   classification label (Sect. 3).
2. Non-lens selection from the official Euclid Q1 catalogues with three rules: R, S and D (Sect. 5).
3. Stamp extraction from the Euclid Q1 MER VIS mosaics (Sect. 4, 6).
4. Verification of all benchmarks (Sect. 7) and characterisation of their separability by simple
   catalogue properties (Sect. 8).

## 2. Data sources

- **Euclid Quick Data Release 1** (Q1; release label `Q1_R1`; public since 19 March 2025; about 63 deg2 at
  the depth of the Euclid Wide Survey; Euclid Collaboration: Aussel et al. 2026). Images: MER
  background-subtracted mosaics; all bands share a grid of 0.1 arcsec pixels; VIS pixel values in ADU/s
  with zero point `MAGZERO` = 24.6 (Q1 Data Product Description Document).
- **Catalogues** in ESA's Euclid Science Archive: `catalogue.mer_catalogue` (positions, total VIS flux
  `flux_detection_total`, `spurious_flag`, `ellipticity`, `segmentation_area`), `catalogue.phz_classification`
  (classification label), `catalogue.mer_morphology` (Sersic fits `sersic_sersic_vis_index`,
  `sersic_sersic_vis_radius`); mosaic list `q1.mosaic_product`. Access: `astroquery.esa.euclid`.
- **Lens candidates:** catalogue of the Euclid Q1 Strong Lensing Discovery Engine
  (`q1_discovery_engine_lens_catalog.csv`; Euclid Collaboration: Walmsley et al. 2026, A&A 711, A26;
  Zenodo doi:10.5281/zenodo.15003116, version v0.0.3).

Two archive column descriptions differ from the data and from the Data Product Description Document:
- `flux_detection_total` is described as an error and `fluxerr_segmentation` as a flux. The values show
  the opposite (lens median 18 uJy and 0.09 uJy respectively); `flux_detection_total` is the total flux.
  AB magnitude = -2.5 log10(flux in uJy) + 23.9.
- `phz_classification` is described as "+1 galaxy, +2 star". The Data Product Description gives
  "+1 star, +2 galaxy, +4 quasar, +8 globular cluster", and the data agree (25.1 million Q1 objects
  with value 2, 0.7 million with value 1). The latter convention is used throughout.

## 3. Lens sample

- Selection: `subset = discovery_engine` (the main search) and expert grade A or B: 497 candidates
  (250 A, 247 B).
- Object identifiers are taken from `id_str` (`<tile>_<identifier>`, minus sign written as `NEG`), which
  stores them exactly; the catalogue's `object_id` column is read as floating point (it has empty rows).
- Euclid classification labels of the 497: 2 (galaxy) 471; 3 (galaxy and star) 15; 0 (no class) 6;
  -1 (unclassified) 2; absent from the table 2; 1 (star) 1.
- **Used: the 487 lenses with a label from 1 to 7** (243 A, 244 B), in all benchmarks.
- Properties of the lens systems (median, 5th-95th percentile): VIS magnitude 20.74 (18.8-22.3);
  segmentation area 1,222 pixels (321-6,911); Sersic index 3.5 (0.6-5.5); Sersic radius 1.1 (0.5-4.0);
  ellipticity 0.22 (0.06-0.51). Catalogue values of lens systems include the light of their arcs.
- Script: `lens_catalogue_properties.py` (writes `lens_catalogue_properties.csv`).

## 4. Stamp format

- 101 x 101 pixels at 0.1 arcsec per pixel (10.1 arcsec), VIS band, the format of the simulated training
  data used in the accompanying work, and comparable to the 10 arcsec cutouts used by the Euclid Q1.
- Centre: the pixel containing the catalogue position (true position within 0.05 arcsec of the centre),
  obtained with `astropy.nddata.Cutout2D` and the world coordinate system of the mosaic.
- Pixel values: as delivered (ADU/s), stored as 32-bit floating point; no rescaling, resampling or filtering.
- Header: WCS, `BUNIT`, `MAGZERO`, `TILEIDX`, `MOSAIC` (full source mosaic name), `OBJTYPE`; lenses
  `LENSID`, `GRADE`, `PHZLABEL`; non-lenses `OBJECTID`, `PHZLABEL`.
- Positions are the catalogue positions, used unchanged; for lenses these are the official Euclid
  positions of the objects chosen by the lens team.

## 5. Non-lens selection

### 5.1 Rules common to R, S and D

- **Classification labels:** R: labels 1-7 (all classified objects); S and D: labels 2 and 3 (galaxy;
  galaxy and star).
- **Quality:** not flagged as a false detection (`spurious_flag = 0`; the flag is set exactly when
  `spurious_prob > 0.5`). Flagged fractions: 6.22% of label-2 objects (1,559,467 of 25,066,388) and
  12.53% of label-3 objects. All lenses have `spurious_prob <= 0.075`. A stricter cut at 0.1 would remove
  about a third of all galaxies, mostly faint ones, and was not used.
- **Exclusions:** no lens candidate of any grade or subset of the lens catalogue (by identifier), and no
  object within 12 arcsec of any candidate (so that no catalogued candidate appears in a non-lens stamp;
  a stamp reaches about 7 arcsec from its centre, and some catalogue positions of candidates outside the
  main search may be off by several arcsec).
- **Reproducibility:** random seed 42. The archive returns query rows in a different order for each
  request; all query results are therefore sorted by object identifier before any random draw (two runs
  give identical selections).
- **Nesting and reserves:** each type has one ordered list; the first 513 non-lenses form the 1k benchmark
  and the first 9,513 the 10k benchmark; 952 further objects (10%) are reserves for replacing objects
  without a complete stamp.

### 5.2 Sampling from the archive

The full catalogue (about 30 million objects) is not read; a fixed pseudo-random subset is selected by
the last digits of the object identifier, which encodes the sky position and is unrelated to brightness
or type:
- R: last four digits below 12 (about 0.12% of all objects): 30,630 objects.
- S and D: last two digits below 10 (about 10%), galaxies brighter than the faintest lens plus
  0.2 magnitudes (VIS < 22.70): 166,809 galaxies.

Representativeness of the R subset: the fraction of its galaxies brighter than each VIS magnitude
from 19 to 25 agrees within 0.1 percentage points with that of all Q1 galaxies (labels 2 and 3, not
flagged), which is 0.34/0.72/1.62/3.80/9.16/22.19/51.98% for magnitudes 19/20/21/22/23/24/25.

Removed as lens candidates and neighbours: R 2 + 54; S 247 + 314; D 247 + 314.

### 5.3 R: random

The R non-lenses are a random draw from the R subset. Final labels of the 10k R non-lenses:
{1: 247, 2: 8,786, 3: 97, 4: 258, 5: 1, 6: 124}.

### 5.4 S: matched in brightness

Motivation: all lenses are brighter than VIS magnitude 23 (median 20.7), whereas only 9.2% of all Q1
galaxies are brighter than 23 and 1.6% brighter than 21. Without matching, brightness
alone would separate lenses from non-lenses.

Method: nearest-neighbour matched sampling in VIS magnitude without replacement. The lenses are visited
in random order (the list is repeated as often as needed); at each visit the not yet selected galaxy
with the closest magnitude is selected. Every lens is visited equally often, so the selected galaxies
reproduce the brightness distribution of the lenses.

Result: magnitude difference to the matched lens median 0.0002, 95% below 0.0018, largest 0.90 (at the
bright end: 38 lenses are brighter than magnitude 19, where candidates are sparse). Percentiles
5/25/50/75/95 of lenses and 10k S non-lenses: 18.76/19.87/20.74/21.60/22.35 for both.

### 5.5 D: matched in brightness, light concentration, size and roundness

Motivation: in S, simple properties of the central object still separate lenses from non-lenses
(Sect. 8): the Sersic index (galaxy type; lenses are mostly early-type galaxies), the ellipticity
(lenses appear rounder) and the size.

Method: as for S, in four properties: VIS magnitude, log10 Sersic index, log10 Sersic radius and
ellipticity (the logarithms because index and radius are positive and span a wide range). Each property
is divided by its standard deviation among the candidates so that all four carry equal weight;
the closest candidate is found with a k-d tree.

Result: 166,248 candidates with valid values; 10,465 selected (labels {2: 10,200, 3: 265}, including
reserves); standardised distance to the matched lens median 0.177, 95% below 0.432, largest 1.61.
Percentiles 5/25/50/75/95, lenses vs 10k D non-lenses:
magnitude 18.76/19.87/20.74/21.60/22.35 vs 18.66/19.85/20.74/21.61/22.34;
Sersic index 0.57/2.03/3.48/5.10/5.50 vs 0.56/1.97/3.46/5.31/5.50;
Sersic radius 0.46/0.74/1.12/1.80/4.02 vs 0.45/0.73/1.06/1.56/3.16;
ellipticity 0.06/0.14/0.22/0.33/0.51 for both. The largest lens systems (radius above about 2 arcsec,
partly enlarged by their arcs) have few equally large look-alikes; 1k D matches more closely than 10k D
(radius percentiles 0.46/0.73/1.10/1.71/3.50).

Scripts: `select_nonlenses.py` (lenses, R, S), `select_nonlenses_D.py` (D).

## 6. Stamp extraction

For each object (`build_vis_benchmarks.py`):
1. All VIS background-subtracted MER mosaics overlapping a 10 arcsec circle around the object are listed
   (`q1.mosaic_product`).
2. The mosaics are tried in turn (for lenses the lens's own tile first, otherwise in order of tile index):
   a region of 8 arcsec radius is downloaded with the archive cutout service and the 101 x 101 stamp is
   cut around the catalogue position. MER tiles overlap; objects near a tile edge are often complete in a
   neighbouring mosaic only, so all covering mosaics are tried before an object is rejected.
3. A stamp is accepted only if it has no pixels outside the mosaic. A technical failure (e.g. a network
   interruption) is repeated once after 5 seconds.
4. Non-lenses without a complete stamp are replaced by the next object of the same list (reserves).

Results: all 487 lens stamps complete. Non-lenses replaced: R 29 of 9,513 (0.3%), S 15 (0.2%), D 17 (0.2%).
Downloads ran four at a time (about 2 hours for R and S together, 76 minutes for D).

## 7. Verification

`verify_vis_benchmarks.py` checks, for all six benchmarks: size (1,000 or 10,000 stamps with 487 lenses);
no duplicate objects; every stamp file exists, has 101 x 101 finite pixels, `BUNIT` = ADU/s, a zero point
and `OBJTYPE` matching the label; the same 487 lenses in all benchmarks; the 1k non-lenses contained in
the 10k non-lenses of the same type; no non-lens is a lens candidate; brightness distributions of S and D
and the Sersic index, Sersic radius and ellipticity of D match the lenses. **Result: all checks passed.**

## 8. Separability by single catalogue properties

To characterise the benchmarks, each stamp is ranked by one catalogue property of its central object and
the area under the ROC curve (AUROC) of that ranking is computed (0.5: the property does not separate
lenses from non-lenses; values far from 0.5 in either direction: it does).

| Property | R (1k / 10k) | S (1k / 10k) | D (1k / 10k) |
|---|---|---|---|
| VIS magnitude | 0.026 / 0.026 | 0.501 / 0.500 | 0.497 / 0.498 |
| Segmentation area (not matched in any type) | 0.978 / 0.980 | 0.581 / 0.588 | 0.498 / 0.502 |
| Sersic index | 0.799 / 0.783 | 0.692 / 0.661 | 0.494 / 0.494 |
| Sersic radius | 0.857 / 0.856 | 0.594 / 0.621 | 0.509 / 0.533 |
| Ellipticity | 0.361 / 0.383 | 0.376 / 0.410 | 0.502 / 0.498 |

R is almost fully separable by brightness or size; S retains a galaxy-type and roundness difference;
D is not separable by any of these properties.

**Reference lens fractions for precision.** The benchmarks contain far more lenses than the survey.
Approximate lens fractions of the underlying populations: about 2 x 10^-4 for galaxies as bright as the
lenses (497 lenses among about 2.2 million Q1 galaxies brighter than VIS magnitude 23) and about
2 x 10^-5 for all classified objects (497 among about 27 million). These are order-of-magnitude values
for extrapolating precision, not survey lens rates.

## 9. Limitations

- Lenses are candidates found by machine learning, citizen scientists and experts; special lenses missed by that
  search are not broadly represented, and grade B candidates are probable, not certain.
- Non-lens status is assumed after excluding all catalogued candidates; a few undiscovered lenses may be
  among the non-lenses.
- Classification labels are photometric and can be wrong for individual objects.
- Catalogue properties of lens systems include their arcs, so matching in D is to whole lens systems.
- Stamps cover 10 arcsec; arcs beyond about 5 arcsec from the centre would be cut (Euclid-detectable
  galaxy-scale lenses mostly have Einstein radii below 3 arcsec).

## 10. Software

Python 3.12; astroquery 0.4.11; astropy 8.0.1; pyvo 1.9.1; numpy 2.5.3; pandas 3.0.6; scipy 1.18.1;
Pillow 12.3.0.

## References

- Euclid Collaboration: Walmsley, M., et al. 2026, *Euclid Quick Data Release (Q1) - XXVI. The Strong
  Lensing Discovery Engine A - System overview and first lens sample*, A&A 711, A26.
- Euclid Collaboration: Lines, N. E. P., et al. 2026, *The Strong Lensing Discovery Engine C - Finding
  lenses with machine learning*, A&A (Euclid Q1 special issue).
- Euclid Collaboration: Aussel, H., et al. 2026, *Euclid Quick Data Release (Q1) - Data release overview*.
- Euclid Q1 Data Product Description Document: https://euclid.esac.esa.int/dr/q1/dpdd/
