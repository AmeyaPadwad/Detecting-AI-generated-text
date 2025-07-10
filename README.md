# 🤖 Detecting AI-Generated Content using Machine Learning

Course Instructor: **Prof. Stratis Ioannidis**  
Course Name: **Machine Learning and Pattern Recognition**  
Course Number: **EECE 5644**  
_Master’s Coursework – Northeastern University_

---

## 📘 Project Overview

With the growing prevalence of generative AI, distinguishing between human-written and machine-generated text has become a critical problem. This project investigates statistical and deep learning models to detect AI-generated content using a subset of the M4 dataset containing 12,000 human-written and 30,000 machine-generated text samples from models like ChatGPT, Bloomz, Dolly, and others.

Our objective was to evaluate and compare three classifiers—**Logistic Regression**, **Random Forest**, and **Neural Networks**—on two types of text features: **Term Frequency (TF)** and **Term Frequency-Inverse Document Frequency (TF-IDF)**.

---

📊 Dataset
We use a subset of the M4 dataset developed by MBZUAI for detecting machine-generated academic text. It contains:

- 12,000 human-written abstracts
- 30,000 AI-generated abstracts from models such as:
  - GPT-3.5 (ChatGPT, Davinci)
  - Bloomz
  - Cohere
  - Dolly
  - FLAN-T5

For this project, we focused on monogram (unigram) features using TF and TF-IDF.

🔗 Dataset Link: https://github.com/mbzuai-nlp/M4

---

## 🧪 Methodology

### 🔍 Preprocessing

- Text cleaning, stop word and punctuation removal
- Tokenization, stemming, and lemmatization using NLTK
- Feature extraction with `CountVectorizer` and `TfidfVectorizer`
- Feature scaling using `StandardScaler`
- Dataset split: **Train\:Valid\:Test = 80:10:10**

### ⚙️ Model Training

Each model was trained and validated with hyperparameter tuning:

- **Logistic Regression**: tuned `C`
- **Random Forest**: tuned `n_estimators`
- **Neural Network (PyTorch)**: tuned `learning rate` and `epochs`

Evaluation metrics:

- ROC AUC Score
- F1 Score
- Accuracy
- Confusion Matrix

---

## 📈 Key Results

| Model               | Features | ROC AUC | F1 Score (Test)   | Accuracy (Test) |
| ------------------- | -------- | ------- | ----------------- | --------------- |
| Logistic Regression | TF       | 0.8796  | 0.7496            | \~75%           |
| Logistic Regression | TFIDF    | 0.8844  | 0.7541            | \~75%           |
| Random Forest       | TF       | 0.9998  | 0.8996            | \~90%           |
| Random Forest       | TFIDF    | 0.9994  | 0.9006            | \~90%           |
| Neural Network      | TF       | 0.9943  | 0.8248 (accuracy) | \~82%           |
| Neural Network      | TFIDF    | 0.9988  | 0.8336 (accuracy) | \~83%           |

🎯 **Random Forest with both TF and TFIDF features outperformed other models** in terms of F1 score and overall robustness.

---

## 🧠 Team Contributions

- **Ameya Padwad** – Feature Engineering, Neural Network modeling
- **Risa Samanta** – Logistic Regression experimentation
- **Balasubramaniam Renganathan** – Random Forest modeling

---

## 📄 Report

See [`Project Report.pdf`](./Project%20Report.pdf) for detailed problem statement, graphs, confusion matrices, and performance metrics.

---
