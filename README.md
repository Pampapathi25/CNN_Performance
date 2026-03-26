# Improving CNN Performance Using Data Augmentation

## Introduction

CNNs automatically extract features from images but can overfit on small datasets. Data augmentation (rotations, flips, zooming, cropping, brightness adjustments) increases dataset diversity and encourages robust feature learning.  

This study evaluates augmentation effects on CNN performance using the CIFAR-10 dataset.

---

## Methodology

- **Dataset:** CIFAR-10 (50,000 train, 10,000 test, 32x32 RGB images, 10 classes)  
- **Models:**  
  - Baseline – original images  
  - Augmented – rotated (±15°), flipped, zoomed (0.9–1.1×) images  
- **CNN Architecture:** 2 conv+pool layers → flatten → FC(128) → softmax output  
- **Training:** 5 epochs, Adam (lr=0.001, batch=32), categorical cross-entropy  
- **Metrics:** Validation accuracy, validation loss  

---

## Results

| Model                  | Accuracy | Loss   |
|------------------------|---------|-------|
| Without Augmentation   | 67.58%  | 0.9484 |
| With Augmentation      | 64.95%  | 0.9927 |

*Observation:* Baseline outperformed the augmented model, suggesting short training and simple architectures may not fully leverage augmentation.

---

## Discussion & Conclusion

- Short training, simple model, and mild augmentation may reduce benefits.  
- Augmentation effectiveness depends on model capacity, training duration, and augmentation strategy.  
- Recommendations: select augmentation wisely, train sufficiently, use capable models, verify label preservation.

---

## Reproducibility

- TensorFlow/Keras preprocessing, augmentation, and training scripts  
- Visualization of accuracy, loss, and augmented images  
- [GitHub Repository Link](#)
