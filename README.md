# Skin Cancer Detection – Deep Learning Research Repository

This repository introduces a comprehensive deep learning framework for automated multi-class Skin Cancer Classification, trained using two large dermoscopic datasets: a binary dataset of 3,297 images and a multi-class dataset of 37,000 images across 14 skin-lesion classes.

The study explores six state-of-the-art architectures combined with four advanced optimizers, including multiple fine-tuning, early stopping, CBAM-attention and ViT transformer-based improvements.

## 📌 Key Research Highlights

| Component | Details |
|-----------|---------|
| Total Images | 3,297 binary + 37,000 multi-class |
| Number of Classes | 14 |
| Deep Learning Models | 6 |
| Optimizers | 4 |
| Evaluation Metrics | Accuracy, F1-score, ROC-AUC |
| Deployment | Python / TensorFlow |

## 📊 Architectures Used

- ✔ InceptionNet-V3
- ✔ MobileNet-V3-Large
- ✔ DenseNet-201
- ✔ EfficientNet-B0
- ✔ DenseNet-151 (P-Band)
- ✔ Vision Transformer ViT-16-S2024

## ⚡ Optimizers Tested

- NADAM
- ADAMW
- ADAMX
- ADAM

## 🔍 Experimental Techniques

- Full transfer learning
- Fine-tuning (layer unfreezing)
- Early stopping
- Learning rate scheduling
- CBAM attention blocks
- Transformer-based ViT
- 50k+ augmentation samples

### Augmentation used:

- rotation
- zoom
- shift
- horizontal flip
- contrast variation

## 📈 Results

### Binary Skin Cancer Detection

| Dataset | Accuracy |
|---------|----------|
| Binary 3,297 images | 97.70% |

### Multi-Class Skin Cancer Detection

| Dataset | Accuracy |
|---------|----------|
| 37,000 images | 92% |
| Best ROC-AUC | 0.93 |

## 🧾 Evaluations Included

- ✔ Class-wise reports
- ✔ Confusion matrix
- ✔ ROC curves
- ✔ Accuracy curve
- ✔ Loss curve
- ✔ Precision / Recall

## 📂 Repository Structure

```
5 model with finetune early stoppage/
5 model with finetune/
5 models with 4 optimizers/
6 models normal training/
CBAM and other/
VIT/
skin cancer checker/
```

Each folder contains:

- training notebook
- saved weights
- results text
- performance output

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- Scikit-learn

**Supports:**

- CPU
- GPU

## 🚀 Usage

### Clone repo

```bash
git clone https://github.com/Kshama-1104/Skin-Cancer-Detection
```

### Install deps

```bash
pip install -r requirements.txt
```

### Run prediction

```bash
python skin_cancer_checker/ultimate_predictor.py
```

## 🤖 Deployment Ready

- Custom prediction utility scripts
- Single-image inference
- Works on real dermoscopic samples
- Supports GPU execution

## 🧠 Research Summary

This research introduces a comprehensive deep learning framework for multi-class Skin Cancer Classification, trained on two datasets: a binary dataset of 3,297 dermoscopic images and a large multi-class dataset of 37,000 images spanning 14 lesion categories. The study evaluates six state-of-the-art architectures—InceptionNet-V3, MobileNet-V3-Large, DenseNet201, EfficientNet-B0, DenseNet151 (P-Band), and ViT-16-S2024—optimized using four advanced optimizers: NADAM, ADAMX, ADAMW, and ADAM.

Extensive augmentation pipelines exceeding 50,000 augmented samples were used to improve robustness and generalization. The framework achieved 97.70% accuracy on the binary dataset and up to 92% accuracy on the 37k multi-class dataset, supported by detailed evaluations including confusion matrices, ROC-AUC analysis (~0.93), and class-wise performance reports.

Overall, this work delivers a scalable, high-accuracy, and clinically relevant pipeline for automated skin cancer detection—offering strong generalization across diverse lesion types and deployment-ready performance on both CPU and GPU environments.

