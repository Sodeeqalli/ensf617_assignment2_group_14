# ENSF 617 – Garbage Classification Programming 
# Team Members:Peter Osaade, Zhanzhi Chen, Sodeeq Alli

# NOTE : COLLABORATION, CODING, MODEL TRAINING AND EVALUATION ALL HAPPENED ON THE TALC CLUSTER
Assignment

This project implements three approaches for garbage classification using the provided CVPR 2024 dataset.

1. Image Model – MobileNetV2 (transfer learning)
2. Text Model – DistilBERT using filename-derived text
3. Multimodal Model – Late fusion combining image and text predictions

All experiments use the dataset splits provided in the assignment.

---

# Repository Structure

image_model/
- train.py
- eval.py
- make_wrong_grid.py
- config.py
- dataset.py
- model.py
- transforms.py
- utils.py

text_model/
- train.py
- eval.py
- config.py
- dataset.py
- model.py
- utils.py

multimodal_model/
- eval_late_fusion.py
- make_multimodal_artifacts.py

image_output_model/

text_output_model/

multimodal_output_model/

Each directory contains the training, evaluation, and supporting utilities for that model.

---

# Models

## Image Model

Backbone: MobileNetV2 (ImageNet pretrained)

Techniques used:
- Transfer learning
- Data augmentation
- Separate learning rates for backbone and classifier
- Confusion matrix and error visualization

Artifacts generated in:

image_output_model/
- best_model.pth
- confusion_matrix.png
- train_val_loss.png
- train_val_acc.png
- wrong_predictions.png

---

## Text Model

Backbone: DistilBERT

Input text is derived from normalized filenames.

Techniques used:
- HuggingFace tokenizer
- Early stopping
- Classification report
- Confusion matrix

Artifacts generated in:

text_output_model/
- best_model.pt
- confusion_matrix.png
- test_summary.txt

---

## Multimodal Model (Late Fusion)

Predictions from both models are combined using:

p_fused = α * p_image + (1 − α) * p_text

The best α is selected by sweeping values on the validation set and then evaluated on the test set.

Artifacts generated in:

multimodal_output_model/
- late_fusion_val_alpha*.txt
- late_fusion_test_alpha*.txt
- confusion_matrix_multimodal_test_alpha*.png
- wrong_predictions_multimodal_test_alpha*.png

---

# Running the Code

Train the models:

python image_model/train.py
python text_model/train.py

Evaluate models:

python image_model/eval.py
python text_model/eval.py

Run multimodal fusion:

python multimodal_model/eval_late_fusion.py

This script:
1. Sweeps α values on the validation set
2. Selects the best value
3. Evaluates on the test set

---

# Results

Model | Test Accuracy
----- | -------------
Image (MobileNetV2) | ~0.75
Text (DistilBERT) | ~0.77
Multimodal Fusion | ~0.78

The multimodal fusion model slightly improves performance over individual models.

---

# Notes

- Model checkpoint files (.pth, .pt) are excluded from Git due to size.
- Evaluation artifacts and logs are included.
- Code is organized modularly for clarity and reproducibility.
