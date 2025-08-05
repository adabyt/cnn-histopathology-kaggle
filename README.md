# CNN for Histopathologic Cancer Detection (Kaggle Practice)

This project uses a **Convolutional Neural Network (CNN)** with **MobileNetV2** as the base model and hyperparameter tuning via **KerasTuner** to classify histopathologic images as **benign** or **metastatic cancer**.

Although this dataset was part of the [Histopathologic Cancer Detection Kaggle Competition](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data), the competition has since closed — this work was done **purely for practice and to expand my deep learning expertise**.

---

## Dataset
- **Source:** [Kaggle - Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data)  
- **Images:** 96x96-pixel patches from lymph node sections.
- **Labels:**  
  - `0`: Benign tissue  
  - `1`: Metastatic cancer  

The dataset was downloaded using the Kaggle API, preprocessed, and split into **training (80%)** and **validation (20%)** sets with stratification to maintain label balance.

---

## Model Architecture
- **Base Model:** MobileNetV2 (ImageNet pretrained, frozen during training)
- **Custom Layers:**
  - GlobalAveragePooling2D
  - Dense layer (tuned units)
  - Dropout layer (tuned rate)
  - Final Dense(1, activation='sigmoid') for binary classification
- **Optimiser & Hyperparameters:** Tuned using **KerasTuner RandomSearch**

---

## Hyperparameter Tuning
Used **KerasTuner Random Search** to find optimal values for:
- Units in Dense layer
- Dropout rate
- Learning rate
- Optimiser choice (`adam` or `rmsprop`)

---

## Results (Best Model)

**Best Hyperparameters:**
- Units in Dense layer: **416**
- Dropout Rate: **0.4**
- Learning Rate: **0.000599**
- Optimiser: **adam**

**Best Model Performance (Validation Set):**
- **Loss:** 0.2956  
- **Accuracy:** 87.57%  
- **Precision:** 87.08%  
- **Recall:** 81.40%  
- **AUC:** 0.9441  
- **F1 Score:** 0.8414  

---

## Project Structure

```
cnn-histopathology-kaggle/
│
├── tme_models/           # Saved models and checkpoints (.keras format)
├── tme_data/             # Dataset directory (Kaggle API download)
├── tme_figures/          # Plots and visualisations
├── logs/                 # TensorBoard logs
├── tme.py                # Main training script
├── .gitignore            # gitignore file
├── requirements.txt      # Specifies files & folders for Git to ignore
├── tme.py                # Main training script
├── tme.ipynb             # Jupyter notebook   
└── README.md             # This file
```

---

## How to Run
1. **Clone the repo**  
```bash
git clone https://github.com/adabyt/cnn-histopathology-kaggle.git
cd cnn-histopathology-kaggle
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Ensure Kaggle API setup
- Place your kaggle.json in ~/.kaggle/
- Accept competition rules on Kaggle

4. Run training script
```
python tme.py
```

---

## Future Improvements
- Fine-tune MobileNetV2 layers (unfreeze top layers for additional training)
- Experiment with other base models (e.g., EfficientNet, ResNet50)
- Try different augmentation strategies for better generalisation
- Expand hyperparameter search space for deeper tuning

---

## Acknowledgements
- Dataset: [Kaggle Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data)
- Pretrained Model: [MobileNetV2](https://arxiv.org/abs/1801.04381)

---

## Note

This project is for educational purposes and deep learning practice — NOT for clinical use.
