# MiscarriageRisk-AI: Machine Learning for ART Outcome Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Sklearn-orange.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🔬 Project Overview

MiscarriageRisk-AI is an advanced machine learning system designed to predict miscarriage risks in Assisted Reproductive Technology (ART) cycles. This project implements state-of-the-art machine learning techniques to analyze medical data and provide risk assessments with 89.90% accuracy.

### 🎯 Key Features

- Advanced preprocessing pipeline for medical data
- Implementation of multiple ML models (Random Forest, SVM, Neural Networks)
- Feature importance analysis and selection
- Handling imbalanced medical data using SMOTE and ADASYN
- Interactive visualization dashboard
- Comprehensive model evaluation metrics

## 🛠️ Technical Architecture

- **Data Processing**: Custom preprocessing pipeline for medical data
- **Feature Engineering**: Advanced feature selection and extraction
- **Model Development**: Ensemble of machine learning models
- **Evaluation**: Comprehensive metrics and visualization tools

## 📊 Results

- Model Accuracy: 89.90%
- Identified 18 key predictive variables
- Balanced handling of minority class cases
- Robust cross-validation results

## 🚀 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/MiscarriageRisk-AI.git

# Install dependencies
pip install -r requirements.txt

# Run the main prediction script
python src/main.py


```markdown
## 📈 Model Performance

<div align="center">

### Overall Model Metrics
| Metric | Score |
|--------|--------|
| Accuracy | 89.90% |
| Precision | 87.65% |
| Recall | 88.73% |
| F1 Score | 88.19% |

### Feature Importance
```python
from matplotlib import pyplot as plt
import seaborn as sns

# Create visualization code here that generates:
# 1. Bar chart of top features
# 2. ROC curve
# 3. Confusion matrix
```

![Model ROC Curve][]

### Confusion Matrix Heatmap```python
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')```

### Training History```python
plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()```
</div>

## 🔗 Project Structure
```
MiscarriageRisk-AI/
├── 📁 data/                  # Data directory (gitignored)
│   ├── 📁 raw/              # Original, immutable data
│   └── 📁 processed/        # Cleaned, transformed data
│
├── 📁 notebooks/            # Jupyter notebooks
│   ├── 📓 01_EDA.ipynb     # Exploratory Data Analysis
│   ├── 📓 02_Features.ipynb # Feature Engineering
│   ├── 📓 03_Models.ipynb   # Model Development
│   └── 📓 04_Eval.ipynb    # Model Evaluation
│
├── 📁 src/                  # Source code
│   ├── 📜 data_prep.py     # Data preprocessing
│   ├── 📜 features.py      # Feature engineering
│   ├── 📜 model.py         # ML model implementation
│   └── 📜 utils.py         # Utility functions
│
├── 📁 tests/               # Unit tests
├── 📁 docs/                # Documentation
└── 📜 README.md           # Project documentation
```

## 📚 Documentation

<div align="center">

### Pipeline Overview```mermaid
graph LR
    A[Raw Data] --> B[Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Evaluation]
    E --> F[Deployment]```

### Key Components
| Component | Description |
|-----------|-------------|
| Data Processing | Handling missing values, outliers, and data normalization |
| Feature Engineering | Selection of 18 key predictive variables |
| Model Architecture | Ensemble of Random Forest, SVM, and Neural Networks |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-Score |

</div>

## 📞 Contact & Social

<div align="center">[![Email](https://img.shields.io/badge/Email-mozr2010%40gmail.com-blue?style=for-the-badge&logo=gmail)](mailto:mozr2010@gmail.com)[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](your-linkedin-url)[![Twitter](https://img.shields.io/badge/Twitter-Follow-blue?style=for-the-badge&logo=twitter)](your-twitter-url)

</div>

## 📖 Citation```bibtex
@article{zare2024miscarriage,
    title={Using Machine Learning to Predict the Risk of Miscarriage 
           in Infertile Couples Undergoing Assisted Reproductive Cycles},
    author={Zare, Mohadese},
    journal={ESHRE 40th Annual Meeting},
    year={2024},
    location={Amsterdam}
}```
```
