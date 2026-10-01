# Datasheet: EUCLID-SAGE black-and-white (VIS) benchmarks

## Motivation

**Purpose.** To evaluate models for discovery of galaxy-scale strong gravitational lenses in real *Euclid* images. 
EUCLID-SAGE provides evaluation benchmarks with expert-graded lens candidates and controlled non-lens populations.

**Gap addressed.** Lens discovery is often reported on non-lens samples that differ from the
lenses in simple properties (brightness, galaxy type), so part of the reported model performance 
can come from those differences rather than from recognising lensing. 

## Composition

**Instances.** Image stamps of astronomical objects, each a 101 x 101 pixel FITS image (Euclid VIS band, 0.1 arcsec per pixel) 
with a binary label: lens (1) or non-lens (0).

**Benchmarks**

| Benchmark | Stamps | Lenses | Non-lenses |
|---|---:|---:|---:|
| EUCLID-SAGE-1k-R_VIS, -S_VIS, -D_VIS | 1,000 each | 487 | 513 |
| EUCLID-SAGE-10k-R_VIS, -S_VIS, -D_VIS (data release) | 10,000 each | 487 | 9,513 |

The same 487 lenses are used in all benchmarks. Non-lenses may appear in more than one benchmark type. 


**Labels**
- *Lenses:* grade A (confident, 243) and grade B (probable, 244) candidates of the main search of the
  Euclid Q1 Strong Lensing Discovery Engine, graded by professional astronomers (Euclid Collaboration:
  Walmsley et al. 2026). 
- *Non-lenses:* objects that are not lens candidates of any grade in that catalogue and lie at least
  12 arcsec from every candidate. Their non-lens status is assumed, not individually verified; a small
  number of undiscovered lenses may be among them (strong lenses are rare: of order one in a few
  thousand bright galaxies).

**Additional information per stamp.** Euclid object identifier, sky position, Euclid classification
label (`phz_classification`), total VIS magnitude, expert grade (lenses), and for S and D non-lenses 
the lens to which they were matched. The lens list adds expert score, Sersic index and radius,
ellipticity and segmentation area.

**Recommended use of the data split.** The benchmarks are test sets. Methods that train or fine-tune on
benchmark images must use cross-validation within one benchmark; because the lenses are shared, a
model trained on one benchmark must not be evaluated on another.

**Errors, noise and redundancy.** Stamps retain the noise characteristics of the Euclid Q1 MER VIS mosaics; 
no denoising or filtering is applied. No stamp has missing pixels. Each object appears once per benchmark (verified).

## Collection process

**Source data.** Euclid Quick Data Release 1 (Q1, release label `Q1_R1`, public since 19 March 2025):
MER background-subtracted VIS mosaics and the catalogues `catalogue.mer_catalogue`,
`catalogue.phz_classification` and `catalogue.mer_morphology`, accessed through ESA's Euclid Science
Archive with `astroquery` (version 0.4.11). Lens candidates: the public catalogue of the Euclid Q1
Strong Lensing Discovery Engine (Zenodo).

**Sampling**
- *Lenses:* all 497 grade A and B candidates of the main search; 487 have a Euclid classification
  label (1–7) and are included; 10 with classification 0, -1 or missing are excluded.
- *Non-lenses:* drawn from a fixed pseudo-random subset of the Q1 catalogue (selected by the last digits
  of the object identifier, which encode sky position and are unrelated to object properties); R at
  random, S and D by nearest-neighbour matching to the lenses (brightness for S; brightness, Sersic
  index, Sersic radius and ellipticity for D). The R subset reproduces the brightness distribution of
  all Q1 galaxies within 0.1 percentage points.

## Preprocessing

Stamps are cut from the mosaics without rescaling, resampling or filtering. Selection quality cuts:
objects flagged by Euclid as likely false detections (`spurious_flag = 1`) are excluded from the
non-lenses; non-lens stamps that could not be completed from any covering mosaic (0.2-0.3% of
non-lenses) were replaced by the next object in the selection order. Display pictures (PNG) can be
produced with `scripts/render_euclid_previews.py`; they are not part of the benchmark data.

## Uses

**Intended.** Evaluating and comparing strong lens discovery on real Euclid data; measuring the gap
between simulated and real performance; studying which lens properties (grade, brightness, galaxy
type) affect discovery.

**Not intended.** Estimating the absolute lens rate or the completeness of a survey search (the lens
fraction of the benchmarks is far higher than in the survey; precision at realistic lens fractions
must be extrapolated as described in the README).

**Known limitations.** The classification label is photometric and can be wrong for individual
objects. The catalogue properties of lens systems include the light of their arcs.

**Confidential or personal data.** None: the data are public astronomical images and catalogues.

## Distribution

The 1k benchmarks are distributed through this repository. The 10k R, S and D benchmarks have been constructed 
and are intended for an archived data release. Planned extensions include colour versions (VIS with NISP Y, J and H) 
using the same objects. Use of Euclid data follows ESA's data credits guide.

**Time frame.** Data taken for Q1; benchmarks constructed in September 2026. 
