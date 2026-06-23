# MSAT Framework

This repository contains the **Multi-Scale Adversarial Training (MSAT)** framework for remote sensing object detection.

MSAT integrates:
- A **SinGAN-based generator** for realistic synthetic augmentation  
- A **Discriminator** for filtering low-quality synthetic samples  
- A **Multi-Scale Attention (MSA/CBAM-based) backbone** for robust feature learning  
- A **hybrid training pipeline** combining real + synthetic data  

---

## ✨ Features

- ✔ SinGAN-based multi-scale synthetic image generation  
- ✔ Realism Discriminator for quality-aware filtering  
- ✔ Multi-Scale Attention (CBAM-enhanced MSAT backbone)  
- ✔ Object detection pipeline for aerial datasets  
- ✔ Supports DOTA, NWPU-VHR10, AID, PatternNet  
- ✔ Evaluation with mAP, Precision, Recall, FID, LPIPS  

---

## 📁 Project Structure
```MSAT/
├── models/
│ ├── singan_msa.py
│ ├── msat.py
│ ├── attention.py
│ ├── discriminator.py
│ └── detector.py
│
├── datasets/
│ ├── dota.py
│ ├── nwpu.py
│ ├── aid.py
│ └── patternnet.py
│
├── evaluation/
│ ├── fid.py
│ ├── lpips.py
│ └── metrics.py
│
├── configs/
│ └── msat.yaml
│
├── train.py
├── test.py
├── generate.py
├── inference.py
└── README.md

```

## ⚙️ Installation

✔ If using GPU (recommended)

Install PyTorch separately based on CUDA:

Example (CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Then install rest:
```bash
pip install -r requirements.txt
```

Recommended:

PyTorch >= 2.0
CUDA-enabled GPU
OpenCV
lpips

## 🚀 Training


Train MSAT using real + synthetic data:
```bash
python train.py
```
Configuration is controlled via:
```bash
configs/msat.yaml
```
## 🧠 Generate Synthetic Data (SinGAN)

Generate realistic remote sensing images:
```bash
python generate.py
```
Outputs are saved to: outputs/generated/

## 🧪 Evaluation

Run evaluation on validation dataset:
```bash
python test.py
```
Metrics include:

mAP@0.5
Precision
Recall
FID
LPIPS

## 🔍 Inference

Run inference on images or folders:
```bash
python inference.py
```
Outputs:

Bounding box visualizations
Saved results in outputs/inference/

## ⚒️ Configuration

All experiments are controlled via:
```bash
configs/msat.yaml
```
Key settings:

Dataset paths
Model backbone settings
SinGAN generator configuration
Training hyperparameters
Evaluation thresholds


## Datasets

Supported datasets:

DOTA → object detection (aerial images)
NWPU-VHR10 → object detection
AID → scene classification
PatternNet → scene classification

## 🤝 Contributions

Contributions are welcome:

New dataset adapters
Improved attention modules
Faster inference optimizations
Better GAN training strategies

## Notes
This framework is research-oriented
Ensure GPU availability for SinGAN generation
For best results, use hybrid training (real + synthetic)


