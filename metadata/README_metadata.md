# Metadata Description

This directory contains the metadata files used to define dataset organization and dataset splits for both the original Scientific Data publication and the Earth Science Informatics study.

## Files

### dataset_labels_original.csv

Metadata file accompanying the original carbonate thin-section dataset published in:

Choi, S.Y., et al. (2026). *High-resolution Annotated Dataset of Girvanella Boundstone Microfacies from the Xiannüdong Formation, China*. Scientific Data.

The file contains the original dataset organization provided in the published dataset, including:

- `dataset_train`
- `dataset_val`
- `dataset_test`

assignments for all image tiles.

---

### dataset_labels_esin_split.csv

Metadata file used in the present Earth Science Informatics study.

This file defines the reorganized dataset structure used for model development and evaluation. The original dataset was reassigned into:

- `train`
- `val`
- `test_mf1`
- `test_mf3`

subsets to enable independent evaluation of the MF1 and MF3 annotation groups.

---

## Relationship Between Metadata Files

The Earth Science Informatics dataset organization was derived directly from the original Scientific Data dataset.

No image content was modified during this process. Only the subset assignments were reorganized.

All image tiles retain their original filenames, allowing direct correspondence between:

- `dataset_labels_original.csv`
- `dataset_labels_esin_split.csv`

---

## Reproducibility

Users can reproduce the dataset organization adopted in this study by:

1. Obtaining the original carbonate thin-section dataset published in Scientific Data.
2. Matching image filenames with the entries in `dataset_labels_esin_split.csv`.
3. Reassigning images according to the subset labels provided in the ESIN metadata file.

The metadata files therefore provide a complete description of the dataset organization used for all analyses presented in the Earth Science Informatics manuscript.