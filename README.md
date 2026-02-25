# 🎭 Deepfake Detection Project (Transformers + Images Only)

A comprehensive deepfake detection system using **Vision Transformers** and **image-only datasets**. This project leverages state-of-the-art transformer architectures to detect manipulated images with high accuracy.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project provides a complete solution for **image-based** deepfake detection using transformer models. It supports training, evaluation, and real-time prediction through a Streamlit web interface.

### What are Deepfakes?

Deepfakes are synthetic media created using artificial intelligence, where a person's face or voice is replaced with someone else's likeness. This project focuses on detecting manipulated **images** using Vision Transformers (ViT), DeiT, and Swin Transformers.

## ✨ Features

- **Transformer-Based Models**: Vision Transformer (ViT), DeiT, and Swin Transformer
- **Image-Only Processing**: Optimized for image datasets (no video processing)
- **Streamlit App**: Well-lit, polished web UI for image upload and detection
- **HuggingFace Integration**: Easy access to image datasets
- **Advanced Training**: Gradient accumulation, warmup scheduling, and early stopping
- **Detailed Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC

## 🚀 Quick Start

### Run the Streamlit App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser. Upload a face image to check if it's real or fake!

**Note**: Works in **demo mode** without a trained model. See [Training](#train-your-own-model) to train your own.

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training, optional for app)
- At least 8GB RAM (16GB+ recommended for training)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Setup HuggingFace (Optional, for HuggingFace datasets)

```bash
pip install huggingface_hub
huggingface-cli login
```

## 📊 Datasets

This project supports **image-only** datasets from HuggingFace and university sources.

### HuggingFace Image Datasets

#### 1. Deepfake Images Dataset (210k images)

**Dataset**: `JamieWithofs/Deepfake-and-real-images-4`

**Dataset Details**:
- **Size**: ~210,000 images
- **Format**: Images with labels (real/fake)
- **Source**: Publicly available deepfake detection dataset

**How to Download**:
```bash
python3 download_datasets.py --dataset JamieWithofs/Deepfake-and-real-images-4
```

Or use the shortcut:
```bash
python3 download_datasets.py --huggingface faceforensics
```

**Dataset Link**: [https://huggingface.co/datasets/JamieWithofs/Deepfake-and-real-images-4](https://huggingface.co/datasets/JamieWithofs/Deepfake-and-real-images-4)

#### 2. Deepfake Images Dataset (190k images)

**Dataset**: `Hemg/deepfake-and-real-images`

**Dataset Details**:
- **Size**: ~190,000 images
- **Format**: Images with labels
- **Source**: Publicly available deepfake detection dataset

**How to Download**:
```bash
python3 download_datasets.py --dataset Hemg/deepfake-and-real-images
```

Or use the shortcut:
```bash
python3 download_datasets.py --huggingface wilddeepfake
```

**Dataset Link**: [https://huggingface.co/datasets/Hemg/deepfake-and-real-images](https://huggingface.co/datasets/Hemg/deepfake-and-real-images)

**Note**: You can also download any other HuggingFace deepfake dataset using:
```bash
python3 download_datasets.py --dataset <dataset-name>
```

## 💻 Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

Features: Images only (JPG, PNG, WebP, BMP), clean UI, demo mode, auto-loads trained models.

### Train Your Own Model

#### Step 1: Download Datasets

```bash
# Download dataset (210k images)
python3 download_datasets.py --dataset JamieWithofs/Deepfake-and-real-images-4

# Or use shortcut
python3 download_datasets.py --huggingface faceforensics
```

#### Step 2: Train

```bash
python3 train.py
```

Trained models will be saved in `models/` directory.

#### Step 3: Run App with Your Model

```bash
streamlit run app.py
```

The app will automatically load the latest model from `models/` directory.

### Evaluate Trained Model

```bash
python3 evaluate.py --model models/best_model_epoch_X.pth
```

## 🏗️ Model Architectures

### 1. Vision Transformer (ViT) - Base

- **Model**: `google/vit-base-patch16-224`
- **Parameters**: ~86M
- **Best For**: General-purpose detection with good balance

### 2. Vision Transformer (ViT) - Large

- **Model**: `google/vit-large-patch16-224`
- **Parameters**: ~307M
- **Best For**: Highest accuracy (requires more GPU memory)

### 3. DeiT (Data-efficient Image Transformer)

- **Model**: `facebook/deit-base-distilled-patch16-224`
- **Parameters**: ~86M
- **Best For**: Efficient training with knowledge distillation

### 4. Swin Transformer

- **Model**: `microsoft/swin-base-patch4-window7-224`
- **Parameters**: ~88M
- **Best For**: Hierarchical feature extraction

## 📈 Results

After training, you'll find results in the `results/` directory:

- `training_history.json`: Complete training metrics
- `training_history.png`: Training curves
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve
- `evaluation_report.txt`: Detailed classification report

### Expected Performance

With proper training on large datasets, you can expect:
- **Accuracy**: 88-96%
- **F1-Score**: 0.88-0.96
- **AUC-ROC**: 0.92-0.98

*Note: Actual performance depends on dataset quality, training duration, and model architecture.*

## 🔧 Configuration

### Model Configuration (`config.py`)

```python
MODEL_CONFIG = {
    "model_name": "vit_base",  # Options: vit_base, vit_large, deit_base, swin_base
    "input_size": 224,
    "num_classes": 2,
    "pretrained": True,
    "dropout": 0.5
}
```

### Training Configuration

```python
TRAIN_CONFIG = {
    "batch_size": 16,          # Smaller batch for transformers
    "num_epochs": 50,
    "learning_rate": 2e-5,     # Lower LR for transformers
    "accumulation_steps": 2,   # Gradient accumulation
    "warmup_steps": 500        # Warmup for transformers
}
```

## 📁 Project Structure

```
Deepfake detection/
├── README.md                 # Documentation
├── requirements.txt          # Dependencies
├── config.py                # Configuration
├── data_loader.py           # Image data loading
├── model.py                 # Transformer models
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── app.py                   # Streamlit app
├── download_datasets.py     # Dataset downloader
├── data/                    # Datasets (auto-created)
│   └── huggingface/
├── models/                  # Model checkpoints
└── results/                 # Training results
```

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Error**
- Reduce `batch_size` in `config.py`
- Increase `accumulation_steps` for gradient accumulation
- Use smaller model (vit_base instead of vit_large)

**2. Dataset Not Found**
- Check dataset paths in `config.py`
- Ensure datasets are downloaded
- Verify file permissions

**3. CUDA Out of Memory**
- Use CPU: Set `device = "cpu"` in config
- Reduce batch size
- Use gradient accumulation

**4. Import Errors**
- Install all dependencies: `pip install -r requirements.txt`
- Make sure transformers is installed: `pip install transformers`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for contribution:
- Additional transformer architectures
- New image dataset integrations
- Performance optimizations
- Documentation improvements
- Bug fixes

## 📝 Citation

If you use this project in your research, please cite the datasets and models you use:

```bibtex
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={ICLR},
  year={2021}
}

@article{rossler2019faceforensics,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={Rössler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nießner, Matthias},
  journal={ICCV},
  year={2019}
}
```

## 📄 License

This project is provided as-is for educational and research purposes. Please check individual dataset licenses before commercial use.

## 🙏 Acknowledgments

- HuggingFace for transformer models and dataset hosting
- Technical University of Munich for FaceForensics++
- Google Research for Vision Transformer
- Facebook AI Research for DeiT
- Microsoft Research for Swin Transformer

---

**Remember**: Deepfake detection is an ongoing research area. Always verify results and consider multiple detection methods for critical applications.

**Last Updated**: February 2026
