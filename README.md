# Reproducing and Extending: "Making Full Use of Probability Maps for Interactive Image Segmentation" (CVPR 2024)

This project reproduces and extends the paper **"MFP: Making Full Use of Probability Maps for Interactive Image Segmentation"** by Lee et al. (CVPR 2024). We implement the original method and introduce a GAN-based extension to improve segmentation performance.

---

## 🔍 Overview

The original paper leverages probability maps to improve interactive image segmentation. In our extension, we propose a GAN-based framework:

- **Generator**: A ResNet-UNet-based `MFPResNetUNet` model outputs binary segmentation masks.
- **Discriminator**: A PatchGAN-style `MFPDiscriminator` differentiates between real and generated masks.

The adversarial setup encourages the generator to produce sharper and more realistic segmentations.

---


---

## 🧪 Datasets

- **Training Dataset**: LVIS (Large Vocabulary Instance Segmentation)
- **Evaluation Dataset**: BSDS500 (Berkeley Segmentation Dataset)

Preprocessing includes resizing, normalization, and transformation for binary segmentation.

---

## ⚙️ Training Configuration

- **Epochs**: 30
- **Loss Functions**:
  - Generator: BCE Loss + Dice Loss
  - Discriminator: Binary Cross-Entropy (PatchGAN loss)
- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Batch Size**: 4

---

## 📊 Results

### Quantitative Metrics (Validation: First 5 Batches)

| Metric      | Score   |
|-------------|---------|
| Dice Score  | 0.7994  |
| IoU         | 0.6659  |
| Precision   | 0.7917  |
| Recall      | 0.8073  |
| Accuracy    | 0.9255  |

### Confusion Matrix (Aggregate over 5 Batches)

- **True Positives**: 194,588  
- **True Negatives**: 1,018,500  
- **False Positives**: 51,190  
- **False Negatives**: 46,442  


The GAN improves segmentation sharpness, but boundary precision still presents challenges.

---

## ⚠️ Limitations

- Thin/irregular structures are sometimes missed.
- Over/under-segmentation occurs in complex backgrounds.
- GAN training adds instability and computational cost.

---

## 💡 Future Work

- Integrate boundary-aware or perceptual loss
- Add CRF-based refinement
- Use attention mechanisms for feature fusion
- Ablate GAN vs non-GAN performance

---
## 👥 Authors

- **Avantika Agarwal** – 202411024  
- **Preet Shah** – 202411053  


