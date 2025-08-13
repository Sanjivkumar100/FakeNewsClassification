# 📰 Fake News Detection using Machine Learning

## 📌 Overview

This project focuses on detecting **fake news** using different machine learning algorithms. The goal is to classify news articles as **REAL** or **FAKE** based on their textual content. Fake news has become a major challenge in today’s digital world, and this project aims to create a model that can assist in filtering out misinformation.

We experimented with multiple classification algorithms and compared their performance based on accuracy, precision, recall, F1-score, and error metrics.

---

## 📂 Dataset

* The dataset contains labeled news articles with the following columns:

  * **text** → The news content
  * **label** → The class (REAL or FAKE)
* Preprocessing steps included:

  * Removing escape sequences and special characters
  * Tokenizing and normalizing text
  * Removing stopwords and punctuation
  * Vectorizing using TF-IDF

---

## ⚙️ Machine Learning Models Used

We tested the following algorithms:

| Algorithm                         | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score (Fake) |
| --------------------------------- | -------- | ---------------- | ------------- | --------------- |
| **Logistic Regression**           | 82.74%   | 0.81             | 0.87          | 0.84            |
| **Passive Aggressive Classifier** | 80.06%   | 0.79             | 0.84          | 0.81            |
| **Naive Bayes**                   | 79.61%   | 0.75             | 0.92          | 0.82            |
| **Support Vector Machine (SVM)**  | 82.66%   | 0.82             | 0.86          | 0.84            |

---

## 📊 Results Summary

* **Best Performing Models:** Logistic Regression and SVM both achieved **\~83% accuracy**.
* **Naive Bayes** showed the highest recall for FAKE news but lower precision, meaning it caught more fake news but also had more false positives.
* **Passive Aggressive Classifier** performed decently but slightly lower than Logistic Regression and SVM.

---

## 🛠️ Tech Stack

* **Language:** Python 🐍
* **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
* **Vectorization:** TF-IDF Vectorizer
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, RMSE, MSE

---

## 📌 Future Improvements

* Use a **larger and more diverse dataset** including multilingual Indian news sources.
* Implement **deep learning models** like LSTMs or BERT for better contextual understanding.
* Build a **real-time fake news detection API** for integration with websites or social media platforms.

---

