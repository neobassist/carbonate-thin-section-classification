# Code Description

This directory contains the source code used for the deep-learning analyses presented in the Earth Science Informatics study:

**"A reproducible informatics workflow for image-based interpretation of carbonate microfacies: a case study from Cambrian calcimicrobial boundstones"**

The scripts provided here were used for model training, prediction generation, and spatial reconstruction of classification results. These workflows operate together with the metadata files provided in the `metadata/` directory and the trained model weights provided in the `weights/` directory.

## Files

### train_efficientnetb3.py

Training workflow for the final EfficientNet-B3 model.

Main functions:

- Dataset loading
- Data preprocessing
- Data augmentation
- EfficientNet-B3 fine-tuning
- Validation monitoring
- MF1/MF3 evaluation
- Export of training diagnostics

---

### generate_predictions.py

Prediction workflow for the trained model.

Main functions:

- Loading trained model weights
- Generating tile-level predictions
- Exporting prediction results as CSV files
- Producing prediction tables for spatial reconstruction

---

### generate_prediction_maps.py

Spatial reconstruction workflow.

Main functions:

- Reading tile-level prediction results
- Reconstructing predictions onto original thin-section images
- Comparing predictions with ground-truth labels
- Generating prediction maps and error-distribution visualizations

This workflow was used to generate the spatial prediction maps presented in Figures 5 and 6 of the manuscript.

## Related Resources

- Dataset metadata: `../metadata/`
- Trained model weights: `../weights/efficientnetb3_final_model.keras`

## Software Environment

The workflows were developed and tested using:

- Python 3.11
- TensorFlow 2.18
- Keras 3.8

## Software Requirements

Required Python packages are listed in:

`../requirements.txt`