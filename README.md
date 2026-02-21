# 📚 Scientific Text Classification using Machine Learning

## 📌 Overview
This project focuses on applying Machine Learning and Natural Language Processing (NLP) techniques to automatically classify scientific articles into three major disciplines:

- 🧪 Chemistry  
- ⚛️ Physics  
- 🧬 Biology  

The classification is based on the **title and abstract** of each scientific article.

The system aims to replace manual classification methods, which are time-consuming and subjective, with an automated and scalable solution.

---

## 🎯 Objectives

- Build a clean and structured scientific text dataset
- Apply NLP preprocessing techniques
- Extract meaningful features using TF-IDF and Bag-of-Words
- Train and compare multiple machine learning models
- Implement Stacking Ensemble to improve performance
- Achieve accuracy ≥ 80% on test data

---

## 🗂 Dataset

- Source: Scientific articles collected via ScienceDirect API
- Fields used:
  - Title
  - Abstract
- Labels:
  - Chemistry
  - Physics
  - Biology

Data preprocessing includes:
- HTML tag removal
- Special character cleaning
- Lowercasing
- Tokenization
- POS Tagging
- Lemmatization

---

## 🧠 Machine Learning Models

### 1️⃣ Naive Bayes (MultinomialNB)
- Probabilistic model based on Bayes Theorem
- Assumes conditional independence between features
- Fast and efficient for text classification

### 2️⃣ Logistic Regression
- Linear classification model
- Uses Softmax for multi-class prediction
- Strong baseline for text data

### 3️⃣ Linear Support Vector Machine (SVM)
- Maximizes margin between classes
- Highly effective in high-dimensional feature spaces
- Performs well with TF-IDF vectors

### 4️⃣ Stacking Ensemble
- Combines:
  - Naive Bayes
  - Logistic Regression
  - Linear SVM
- Uses Logistic Regression as meta-model
- Applies K-Fold Cross Validation (K=5)
- Improves generalization performance

---

## ⚙️ Feature Engineering

- Bag-of-Words (BoW)
- TF-IDF Vectorization

These techniques convert raw text into numerical vectors suitable for ML models.

---

## 📊 Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Target performance:
- ≥ 80% Accuracy
- Balanced Precision & Recall across 3 classes

---

## 🏗 Project Structure
