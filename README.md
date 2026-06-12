# carbonate-thin-section-classification

This repository contains the source code, metadata files, trained model weights, and supplementary resources associated with the Earth Science Informatics study:

**"A reproducible informatics workflow for image-based interpretation of carbonate microfacies: a case study from Cambrian calcimicrobial boundstones"**

This study builds upon the openly available carbonate thin-section dataset published in *Scientific Data* (Choi et al., 2026).

The repository provides the complete workflow used in the study, including model training, prediction generation, spatial reconstruction of classification maps, dataset metadata, and the final EfficientNet-B3 model weights.

The study evaluates classification performance across two independently annotated microfacies groups (MF1 and MF3) and investigates the influence of dataset composition and annotation context on model behavior.

## Repository Structure

```text
repository/

README.md
LICENSE
CITATION.cff
requirements.txt

code/
├── train_efficientnetb3.py
├── generate_predictions.py
├── generate_prediction_maps.py
└── README_code.md

metadata/
├── dataset_labels_original.csv
├── dataset_labels_esin_split.csv
└── README_metadata.md

weights/
└── efficientnetb3_final_model.keras
```

## Repository Contents

- **code/** — Source code used for model training, prediction generation, and spatial reconstruction of classification maps.
- **metadata/** — Metadata files defining both the original Scientific Data dataset split and the reorganized ESIN dataset split.
- **weights/** — Final trained EfficientNet-B3 model weights used in the study.
- **requirements.txt** — Python package requirements used for the analyses.

The file `efficientnetb3_final_model.keras` contains the final model weights used to generate all results presented in the manuscript, including classification metrics, confusion matrices, and spatial reconstruction maps.

Detailed descriptions of the source code and metadata files are provided in:

- `code/README_code.md`
- `metadata/README_metadata.md`

## Citation

If you use this repository, please cite the associated Earth Science Informatics article:

Choi, S.Y., et al. (2026). *A reproducible informatics workflow for image-based interpretation of carbonate microfacies: a case study from Cambrian calcimicrobial boundstones*. Earth Science Informatics.

The original carbonate thin-section dataset was published separately and should also be cited when the dataset is used:

Choi, S.Y., Kim, D.C., Hong, J., Lee, B.G., Do, J.D., Kim, C.H., & Lee, C.W. (2026). *High-resolution Annotated Dataset of Girvanella Boundstone Microfacies from the Xiannüdong Formation, China*. Scientific Data, 13, 611. https://doi.org/10.1038/s41597-026-06958-1

## License

This repository is distributed under the MIT License. See the `LICENSE` file for details.