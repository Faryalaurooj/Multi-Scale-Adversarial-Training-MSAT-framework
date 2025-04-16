# MSAT Framework

This repository contains the Multi-Scale Adversarial Training (MSAT) framework for enhancing object detection in remote sensing via synthetic + real hybrid training.

### Features

- SinGAN-based realistic augmentation
- Realism Discriminator
- Multi-Scale Attention (MSA)
- Support for hybrid datasets
- Evaluation across multiple real-world datasets

### Directory Structure

MSAT-Framework/
├── main.py                  
├── train_gan.py
├── dataset.py
├── config.yaml
├── README.md
├── utils/
│   └── singan_wrapper.py
├── models/
│   ├── discriminator.py
│   └── msa_module.py
├── data/
│   ├── real/
│   └── synthetic/
├── outputs/
│   └── generated_samples/


COMMANDS :

### Run

```bash
python main.py
