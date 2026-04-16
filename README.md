<h1 align="center"> Alzheimer Disease Detection using Transfer Learning</h1>

<p align="center">
  <strong>A deep learning system that classifies brain MRI scans into three Alzheimer's disease stages using transfer learning with pre-trained CNN architectures (MobileNet, VGG, InceptionV3).</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
  <a href="#"><img src="https://img.shields.io/badge/Transfer%20Learning-Pretrained-blue?style=for-the-badge" alt="Transfer Learning"></a>
  <a href="#"><img src="https://img.shields.io/badge/Category-Medical%20AI-red?style=for-the-badge" alt="Medical AI"></a>
  <a href="#"><img src="https://img.shields.io/badge/Course-CO324%20Project-purple?style=for-the-badge" alt="Course"></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-medical-context">Medical Context</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-approach">Approach</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-results">Results</a>
</p>

---

## 📖 Overview

A **deep learning project** that applies **transfer learning** with pre-trained convolutional neural networks (CNNs) to classify brain MRI scans into three Alzheimer's Disease stages. By leveraging ImageNet-pretrained architectures, the model achieves strong performance even with limited medical imaging data.

This project was developed as a **CO324 semester project** and explores how well general-purpose pre-trained CNN architectures transfer to specialized medical imaging tasks.

> **Why this matters:** Alzheimer's disease affects over 55 million people worldwide. Early detection from MRI scans can enable timely intervention and significantly improve patient outcomes.

---

## 🏥 Medical Context

### What is Alzheimer's Disease?

Alzheimer's Disease (AD) is a progressive neurodegenerative disorder that affects memory, thinking, and behavior. It's the most common cause of dementia, affecting millions of people globally.

### Stages of Cognitive Decline

| Stage | Abbreviation | Description |
|-------|:------------:|-------------|
| **Cognitively Normal** | CN | Healthy individuals with no cognitive impairment |
| **Cognitive Impairment** | CI / MCI | Mild cognitive impairment - early warning signs |
| **Alzheimer's Disease** | AD | Confirmed Alzheimer's with significant cognitive decline |

### Why MRI?

Magnetic Resonance Imaging (MRI) reveals structural changes in the brain associated with Alzheimer's - particularly **hippocampal atrophy** and **ventricular enlargement**. Deep learning models can detect these subtle patterns that may be difficult for the human eye.

---

## 📊 Dataset

The project uses brain MRI images organized into three classes:

```
diseases/
├── AD/          # Alzheimer's Disease scans
├── CI/          # Cognitive Impairment scans
└── CN/          # Cognitively Normal scans
```

Each class folder contains MRI images (also provided as ZIP archives: `AD.zip`, `CI.zip`, `CN.zip` for easy distribution).

The dataset follows the standard **ADNI (Alzheimer's Disease Neuroimaging Initiative)** classification convention, which is widely used in medical AI research.

---

## 🎯 Approach

### Transfer Learning

Rather than training a CNN from scratch (which requires massive datasets), this project uses **transfer learning** - leveraging models pre-trained on ImageNet as feature extractors.

**Why transfer learning works here:**
- Pre-trained models already know how to detect generic visual features (edges, textures, shapes)
- Medical imaging datasets are typically small - training from scratch would overfit
- Fine-tuning is dramatically faster than training from scratch
- Achieves higher accuracy with limited data

### Models Compared

| Model | Strengths | Trade-offs |
|-------|-----------|------------|
| **MobileNet** | Lightweight, fast inference, mobile-friendly | Slightly lower accuracy than larger models |
| **VGG** | Simple, deep architecture with strong feature extraction | Large model size, slower training |
| **InceptionV3** | Multi-scale feature extraction via inception modules | More complex architecture |

### Training Pipeline

```
MRI Images (224×224)
        │
   Data Augmentation (rotation, zoom, flip)
        │
   Pre-trained CNN (frozen base layers)
        │
   Custom Classification Head (Dense layers)
        │
   Softmax Output (3 classes: AD / CI / CN)
```

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.7+ | Core implementation |
| **Deep Learning** | TensorFlow / Keras | Model architecture and training |
| **Transfer Learning** | ImageNet pre-trained weights | Feature extraction backbone |
| **Image Processing** | Keras `ImageDataGenerator` | Data loading and augmentation |
| **Notebook** | Jupyter | Interactive development |
| **Visualization** | Matplotlib | Training curves and results |

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Input Image Size** | 224 × 224 × 3 |
| **Pre-trained Weights** | ImageNet |
| **Epochs** | 20 |
| **Number of Classes** | 3 (AD, CI, CN) |
| **Output Activation** | Softmax |

---

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- GPU recommended for training (CPU works but slower)

### Setup

```bash
# Clone the repository
git clone https://github.com/zishnusarker/Alzheimer-Detection.git
cd Alzheimer-Detection

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📋 Usage

### 1. Extract the Dataset

```bash
cd diseases
unzip AD.zip
unzip CI.zip
unzip CN.zip
```

### 2. Launch the Notebook

```bash
jupyter notebook pr-final-project.ipynb
```

### 3. Run Through the Cells

The notebook walks through:
1. **Library imports** - TensorFlow, Keras, NumPy, Matplotlib
2. **Data loading** - Loading MRI images with `ImageDataGenerator`
3. **Data augmentation** - Rotation, flip, zoom for robustness
4. **Model building** - Pre-trained backbone + custom classification head
5. **Training** - 20 epochs with Adam optimizer
6. **Evaluation** - Accuracy, loss curves, model comparison
7. **Prediction** - Classify new MRI scans

---

## 📈 Results

The project generates several visualizations (see `code ss/` folder):

- **Model Architecture** - Visual diagram of the transfer learning setup
- **Accuracy & Loss Curves** - Training and validation metrics across epochs
- **Final Evaluation** - Test set accuracy and classification metrics
- **Prediction Samples** - Model predictions on unseen MRI scans

### Sample Outputs

The `code ss/` and `other ss/` folders contain:
- Code execution screenshots
- Training accuracy graphs
- Model architecture visualizations
- Comparative performance analysis
- Transfer learning concept illustrations

---

## 📁 Project Structure

```
Alzheimer-Detection/
├── README.md                                # Project documentation
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignore rules
├── CO324 Project Report.doc                 # Full academic project report
├── pr-final-project.ipynb                   # Main Jupyter notebook
│
├── diseases/                                # Dataset (3 classes)
│   ├── AD/                                  # Alzheimer's Disease MRI images
│   ├── AD.zip                               # Compressed AD dataset
│   ├── CI/                                  # Cognitive Impairment MRI images
│   ├── CI.zip                               # Compressed CI dataset
│   ├── CN/                                  # Cognitively Normal MRI images
│   └── CN.zip                               # Compressed CN dataset
│
├── code ss/                                 # Code execution screenshots
│   ├── architecture of the model.png
│   ├── accuracy graph.png
│   ├── accuracy and loss.png
│   ├── model evaluate.png
│   ├── final result evaluate.png
│   └── s1.png, s2.png, ... s13.png
│
└── other ss/                                # Conceptual diagrams & references
    ├── Transfer Learning with Pre-trained Deep Learning Models.png
    └── graph Performance of off-the-shelf pre-trained models.png
```

---

## 🎓 Key Concepts Demonstrated

<details>
<summary><strong>What is Transfer Learning?</strong></summary>

Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a second task. In this project, CNNs pre-trained on ImageNet (1.2 million natural images, 1000 classes) are adapted to classify brain MRI scans - even though MRI images look nothing like the original training data, the low-level visual features (edges, textures, shapes) learned by these models transfer remarkably well.

</details>

<details>
<summary><strong>Why freeze pre-trained layers?</strong></summary>

The early layers of a CNN learn generic features (edges, colors, textures). These are useful for any image classification task, so we freeze them to preserve what they've learned. We only train the final classification layers on our specific task (Alzheimer's detection), which prevents overfitting and dramatically reduces training time.

</details>

<details>
<summary><strong>Why 224×224 input size?</strong></summary>

This is the standard input size for most ImageNet pre-trained models (VGG, ResNet, MobileNet, InceptionV3). Using this size lets us use the pre-trained weights directly without modification. The original MRI images are resized to match this dimension.

</details>

<details>
<summary><strong>Why is data augmentation important in medical imaging?</strong></summary>

Medical imaging datasets are typically small (hundreds to thousands of images, vs millions in ImageNet). Data augmentation - rotating, flipping, zooming, shifting - artificially expands the dataset by creating variations of existing images. This helps the model generalize better and reduces overfitting.

</details>

<details>
<summary><strong>Why Softmax activation in the output layer?</strong></summary>

Softmax converts raw model outputs into probabilities that sum to 1.0 across all classes. For multi-class classification (AD vs CI vs CN), it gives us interpretable probabilities like "70% AD, 20% CI, 10% CN" for each scan.

</details>

---

## 📚 Academic Context

This project was developed as part of **CO324 (Machine Learning / Pattern Recognition)** coursework. The full academic report is available as `CO324 Project Report.doc` in the repository, covering:

- Literature review of Alzheimer's detection approaches
- Dataset description and preprocessing
- Model architecture and training methodology
- Experimental results and analysis
- Comparison with existing approaches
- Conclusions and future work

---

## 🔮 Future Improvements

- Use larger, more diverse datasets (ADNI, OASIS)
- Implement ensemble methods combining multiple pre-trained models
- Add explainability with Grad-CAM to show what regions the model focuses on
- Experiment with 3D CNNs for volumetric MRI analysis
- Deploy as a web app using Flask/Streamlit for clinical demonstration
- Add confusion matrix, ROC curves, and detailed classification reports
- Experiment with newer architectures (EfficientNet, Vision Transformer)
- Fine-tune the entire network after initial transfer learning phase
- Cross-validation for more robust accuracy estimation

---

## ⚠️ Medical Disclaimer

> **This project is for educational and research purposes only.** It is **not** a diagnostic tool and must **not** be used for actual medical diagnosis or treatment decisions. Alzheimer's diagnosis requires comprehensive clinical evaluation by qualified medical professionals, including neurological exams, cognitive testing, and additional imaging studies. Always consult healthcare providers for medical decisions.

---

## 📄 License

This project is available under the MIT License.

---

<p align="center">
  Made with ❤️ for healthcare AI education
</p>

<p align="center">
  <strong>Applying deep learning to early Alzheimer's detection 🧠</strong>
</p>
