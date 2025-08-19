
# GAN Implementation for Tabular Data Generation: Adult Dataset Analysis

## 📌 Project Overview

This project implements and evaluates **Generative Adversarial Networks (GANs)** for synthetic tabular data generation using the **Adult Census dataset**.
We developed two architectures:

* **Standard GAN**
* **Conditional GAN (cGAN)**

Both models leverage an **autoencoder-based architecture** to handle the dataset’s **mixed numerical and categorical features**.
While the cGAN demonstrated **superior performance in predictive utility**, both models struggled with **categorical feature discretization**, resulting in synthetic data that remains **easily distinguishable from real data** (AUC = 1.0000).

---

## 📂 Dataset

* **Adult Census Dataset** (32,561 samples, 14 features, 1 binary target: income ≤50K or >50K)
* **Features**:

  * 6 numerical: `age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week`
  * 8 categorical: `workclass, education, marital-status, occupation, relationship, race, sex, native-country`
  * Target distribution: **75.9% ≤50K**, **24.1% >50K**

### Preprocessing

1. Normalize numerical features with **StandardScaler**
2. Encode categorical features with **Label Encoding**
3. Binary encode target labels (`≤50K → 0`, `>50K → 1`)
4. Stratified train-test split (80/20)
5. Multi-seed evaluation with seeds `[42, 123, 456]`

---

## 🏗️ Architecture

### Autoencoder Backbone

* Input: 14D → Embedding: 32D → Reconstructed: 14D
* Purpose: Provide a unified continuous embedding space for GAN training.

### Standard GAN

* **Generator**: Noise(100D) → \[128 → 256 → 128] → Embedding(32D)
* **Discriminator**: Embedding(32D) → \[128 → 64 → 32] → Real/Fake

### Conditional GAN

* Conditioning on **income label**
* **Generator**: Noise(100D) + Label(2D) → Embedding(32D)
* **Discriminator**: Embedding(32D) + Label(2D) → Real/Fake

---

## ⚙️ Training Configuration

* Autoencoder: 100 epochs, **MSE loss**, Adam(lr=0.0002)
* GANs: 200 epochs, **BCE loss**, Adam(lr=0.0002)
* Batch size: 64
* Device: CUDA-enabled GPU

**Training Challenges & Fixes**:

* Discriminator overpowered → applied **label smoothing**, tuned network size, adjusted learning rates.
* Final discriminator accuracy balanced at \~83.6% (improved from 90%+).

---

## 📊 Results

### 🔹 Data Quality (Detection Metric)

* Random Forest classifier trained on **real + synthetic**
* Result: **AUC = 1.0000** (synthetic easily detected due to continuous outputs for categorical features).

### 🔹 Predictive Utility (Efficacy Metric)

* Random Forest trained on **synthetic data**, tested on **real data**
* **Standard GAN efficacy**: 0.4786 ± 0.0458
* **Conditional GAN efficacy**: 0.6541 ± 0.0702

✅ Conditional GAN preserved **17.6% more predictive utility** than Standard GAN.

### 🔹 Feature Distributions & Correlations

* **Standard GAN**:

  * Failed to maintain discrete structure (e.g., fractional education years).
  * Unrealistic work-hour patterns.
* **Conditional GAN**:

  * Better discrete feature preservation.
  * More realistic employment and education patterns.
  * Superior correlation preservation between features.

---

## 📌 Key Insights

* **Conditional GAN > Standard GAN** in:

  * Predictive utility
  * Feature distribution realism
  * Correlation preservation
* **Both models struggle** with:

  * Handling categorical features properly
  * Perfect detectability by simple classifiers
  * Substantial predictive utility loss vs real data

---

## 📖 References

* UCI Adult Census Dataset
* GANs for Tabular Data (CTGAN, medGAN, etc.)
* Goodfellow et al., “Generative Adversarial Nets” (2014)

---

## 🛠️ Getting Started

### Requirements

```bash
python >= 3.8
torch >= 1.10
scikit-learn
numpy
pandas
matplotlib
```
