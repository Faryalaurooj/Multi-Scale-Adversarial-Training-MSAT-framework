
# MSAT Framework

This repository contains the **Multi-Scale Adversarial Training (MSAT)** framework for enhancing object detection in remote sensing through a combination of real and synthetically augmented data.

MSAT integrates a SinGAN-based generator for creating high-fidelity augmentations, a Realism Discriminator for assessing data quality, and a Multi-Scale Attention (MSA) module to improve feature relevance during detection.

![hybrid](https://github.com/user-attachments/assets/f8784eda-2e4c-4a2b-99ad-3b2e7cae9fff)
![flowdiagram_complete](https://github.com/user-attachments/assets/fd1fcbe4-37b6-4760-9f8d-b32f41d542b4)




### ✨ Features

- ✅ SinGAN-based realistic augmentation of remote sensing imagery  
- ✅ Realism Discriminator for filtering low-quality synthetic data  
- ✅ Multi-Scale Attention (MSA) for enhanced feature extraction  
- ✅ Compatible with hybrid (real + synthetic) datasets  
- ✅ Evaluated on multiple real-world aerial/satellite datasets  



Here's how you can use your MSAT framework code step-by-step. These are command-line instructions for **training, generating synthetic data, evaluating**, and automating the adaptive pipeline.

---

### 🛠️ 1. **Install Requirements**
Install dependencies (create a virtual environment if needed):

```bash
pip install -r requirements.txt
```

Make sure `PyTorch`, `OpenCV`, and compatible `CUDA` versions are installed.

---

### 📁 2. **Folder Structure Overview**
```
MSAT_Framework/
├── train.py              # For training the detection model
├── generate.py           # For generating synthetic images with SinGAN
├── evaluate_image.py     # For evaluating and adapting harmonization
├── configs/
│   ├── yolov10.yaml      # YOLOv10 training config
│   └── singan.yaml       # SinGAN generation config
├── weights/              # Store trained weights
├── data/
│   ├── real/             # Real training images
│   ├── hybrid/           # Folder for synthetic data
│   └── annotations/      # Corresponding label files
├── models/               # Detection and generation models
└── utils/                # Helper functions
```

---

### 🚀 3. **Train Detection Model**

You can train on real or hybrid data:

```bash
python train.py \
  --config configs/yolov10.yaml \
  --data data/real \
  --weights weights/yolov10.pt \
  --epochs 100 \
  --batch-size 16
```

---

### 🧠 4. **Generate Synthetic Data using SinGAN**

This will generate and harmonize synthetic aircraft images and place them into `data/hybrid`.

```bash
python generate.py \
  --config configs/singan.yaml \
  --output_dir data/hybrid \
  --num_samples 100 \
  --harmonization-scale 1.0
```

---

### ✅ 5. **Evaluate with Real-Trained Model and Adapt Harmonization**

Evaluates whether the model trained on real data detects objects in synthetic images. If not, it adjusts harmonization scale and retries.

```bash
python evaluate_image.py \
  --model weights/yolov10.pt \
  --synthetic_dir data/hybrid \
  --real_data_dir data/real \
  --harmonization_step 0.1 \
  --max_attempts 5
```

---

### 🔁 6. **Fully Automated Pipeline**

To combine generation, training, and evaluation in a loop, create a shell or Python script that calls:
```bash
generate.py → train.py → evaluate_image.py → repeat if needed
```

Let me know if you want me to prepare that automation script too.

---

Would you like:
- A `run_pipeline.sh` or `pipeline.py` file to do this loop for you?
- GitHub `README.md` file to help you document it?



### ⚒️ Configuration

Edit `config.yaml` to control:

- Dataset paths
- Training parameters
- GAN and Discriminator options
- MSA module settings



### 📊 Results

MSAT has been evaluated on several aerial datasets. It demonstrates improved F1-scores and detection robustness over baseline YOLOv8 and other models, especially when trained with hybrid datasets.

| Model     | Dataset       | F1-Score | Precision | Recall |
|-----------|---------------|----------|-----------|--------|
| YOLOv10    | DOTA         | 0.72     | 0.70      | 0.74   |
| **MSAT**  | DOTA       | **0.82** | **0.80**  | **0.85** |
| MSAT + MSA | Custom Hybrid | **0.87** | **0.86**  | **0.89** |



### 🧪 Dataset Notes

- Real images are stored under `data/real/`
- Synthetic images generated by SinGAN are stored in `data/synthetic/`
- You can mix datasets by modifying the loader in `dataset.py`



### 🤝 Contributions

Feel free to fork the repo and submit pull requests for:

- New dataset adapters
- Performance improvements
- Integration with other backbones (YOLOv9, Faster-RCNN)



