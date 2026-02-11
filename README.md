# Deepfake Image Detection using Xception-CBAM

Author

**Mustafa Genji**
*Deepfake Image Detection based on Convolutional Neural Networks*

---

## Overview

This repository presents a deep learning framework for detecting deepfake facial images using an enhanced Xception convolutional neural network integrated with a Convolutional Block Attention Module (CBAM) and a custom classification head.

The goal is to improve feature representation by guiding the model to focus on spatial and channel-level forgery artifacts commonly present in AI-generated faces.

---

## Method

The proposed model combines:

* Xception backbone network (transfer learning)
* CBAM attention mechanism
* Custom classification head
* Data augmentation
* Binary classification (Real vs Fake)

The CBAM module helps the network focus on important forgery regions such as:

* blending boundaries
* texture inconsistencies
* abnormal facial patterns

---

## Dataset

We use the **140K Real vs Fake Faces Dataset** from Kaggle:

https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

Due to size limitations, the dataset is NOT included in this repository.

Follow instructions in:

```
dataset/README_dataset.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/YOURUSERNAME/Deepfake-Image-Detection-Xception-CBAM.git
cd Deepfake-Image-Detection-Xception-CBAM
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Training

After downloading and extracting the dataset:

```
python training/train.py
```

The best model will be saved as:

```
best_model.keras
```

---

## Evaluation

```
python training/evaluate.py
```

The script prints classification metrics including:

* Accuracy
* Precision
* Recall
* F1-Score

---

## Project Structure

```
Deepfake-Image-Detection-Xception-CBAM/
│
├── models/           # Model architecture
├── training/         # Training and evaluation scripts
├── dataset/          # Dataset instructions only
├── notebooks/        # Experiments
└── results/          # Figures and metrics
```

---

## Research Purpose

This project was developed as part of a Master's thesis focused on detecting manipulated facial images using convolutional neural networks and attention mechanisms.

---

## Citation

If you use this code in your research, please cite:

@mastersthesis{genji2025deepfake,
title={Deepfake Image Detection based on Convolutional Neural Networks},
author={Genji, Mustafa},
year={2025}
}

---

## License

This project is licensed under the MIT License.
