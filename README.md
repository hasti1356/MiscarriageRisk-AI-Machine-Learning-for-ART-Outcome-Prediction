# MiscarriageRisk-AI: Machine Learning for ART Outcome Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Sklearn-orange.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)


```markdown
# 🔬 MiscarriageRisk-AI

## 📊 Project Overview

MiscarriageRisk-AI is an advanced machine learning system designed to predict miscarriage risks in Assisted Reproductive Technology (ART) cycles. This project implements state-of-the-art machine learning techniques to analyze medical data and provide risk assessments with 89.90% accuracy.

## 🎯 Key Features

- Advanced preprocessing pipeline for medical data
- Implementation of multiple ML models (Random Forest, SVM, Neural Networks)
- Feature importance analysis and selection
- Handling imbalanced medical data using SMOTE and ADASYN
- Interactive visualization dashboard
- Comprehensive model evaluation metrics

## 🛠️ Technical Architecture

| Component | Details |
|-----------|---------|
| Data Processing | Custom preprocessing pipeline for medical data |
| Feature Engineering | Advanced feature selection and extraction |
| Model Development | Ensemble of machine learning models |
| Evaluation | Comprehensive metrics and visualization tools |

## 📊 Results & Metrics

| Metric | Score |
|--------|--------|
| Accuracy | 89.90% |
| Precision | 87.65% |
| Recall | 88.73% |
| F1 Score | 88.19% |

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MiscarriageRisk-AI.git

# Install dependencies
pip install -r requirements.txt

# Run the main prediction script
python src/main.py
```

## 📁 Project Structure

```
MiscarriageRisk-AI/
├── data/                  # Data directory
│   ├── raw/              # Original data
│   └── processed/        # Cleaned data
│
├── notebooks/            # Jupyter notebooks
│   ├── 01_EDA.ipynb     
│   ├── 02_Features.ipynb
│   ├── 03_Models.ipynb  
│   └── 04_Eval.ipynb    
│
├── src/                  # Source code
│   ├── data_prep.py     
│   ├── features.py      
│   ├── model.py         
│   └── utils.py         
│
├── tests/               # Unit tests
├── docs/                # Documentation
└── README.md           
```

## 🔍 Key Components

<table>
  <tr>
    <th>Component</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><b>🔧 Data Processing</b></td>
    <td>
      • Missing value imputation<br>
      • Outlier detection and handling<br>
      • Data normalization and scaling<br>
      • Data quality validation
    </td>
  </tr>
  <tr>
    <td><b>⚡ Feature Engineering</b></td>
    <td>
      • Selection of 18 key predictive variables<br>
      • Feature extraction and transformation<br>
      • Dimensionality reduction<br>
      • Feature importance analysis
    </td>
  </tr>
  <tr>
    <td><b>🤖 Model Architecture</b></td>
    <td>
      • Random Forest Classifier<br>
      • Support Vector Machine<br>
      • Neural Networks<br>
      • Ensemble Method Integration
    </td>
  </tr>
  <tr>
    <td><b>📊 Evaluation Metrics</b></td>
    <td>
      • Accuracy: 89.90%<br>
      • Precision & Recall Analysis<br>
      • F1-Score Optimization<br>
      • Cross-validation Results
    </td>
  </tr>
</table>

## 📞 Contact Information

<table>
  <tr>
    <td align="center">📧</td>
    <td><a href="mailto:mozr2010@gmail.com">mozr2010@gmail.com</a></td>
  </tr>
  <tr>
    <td align="center">🎓</td>
    <td>ESHRE 40th Annual Meeting Presenter</td>
  </tr>
</table>

## 📖 Citation

```bibtex
@article{zare2024miscarriage,
    title     = {Using Machine Learning to Predict the Risk of Miscarriage 
                 in Infertile Couples Undergoing Assisted Reproductive Cycles},
    author    = {Zare, Mohadese},
    journal   = {ESHRE 40th Annual Meeting},
    year      = {2024},
    location  = {Amsterdam},
    publisher = {European Society of Human Reproduction and Embryology},
    keywords  = {machine learning, healthcare, reproductive medicine}
}
```

<div align="center">
  <kbd>Status: Active</kbd> &nbsp; <kbd>Version: 1.0.0</kbd> &nbsp; <kbd>Updated: 2024</kbd>
</div>
```


<div align="center">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge&logo=github"/>
  <img src="https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge&logo=semantic-release"/>
</div>
```
