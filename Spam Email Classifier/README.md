# 📧 Spam Email Classifier using Machine Learning

A Machine Learning project that classifies emails as **Spam** or **Not Spam (Ham)** using the **Multinomial Naive Bayes** algorithm. The model is trained on a preprocessed email dataset and learns to distinguish spam emails based on word frequency features.

---

## 🚀 Features

- Detects Spam and Non-Spam emails
- Uses the Multinomial Naive Bayes classification algorithm
- Train-test split for model evaluation
- Generates classification metrics
- Displays a confusion matrix for performance analysis
- Simple and efficient machine learning pipeline

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook (Kaggle)

---

## 📂 Project Structure

```
Spam-Email-Classifier/
│
├── spam-email-classification.ipynb
├── README.md
├── requirements.txt
└── images/
    └── confusion_matrix.png
```

---

## 📊 Dataset

This project uses a preprocessed **Email Spam Classification Dataset** containing numerical word-frequency features.

**Dataset Features**
- 3000+ word frequency columns
- Email identifier
- Target label (`Prediction`)
  - **0 → Not Spam (Ham)**
  - **1 → Spam**

---

## ⚙️ Workflow

```
Email Dataset
      │
      ▼
Load Dataset
      │
      ▼
Feature Selection
(Remove Email ID)
      │
      ▼
Train-Test Split
      │
      ▼
Multinomial Naive Bayes
      │
      ▼
Prediction
      │
      ▼
Performance Evaluation
      │
      ▼
Confusion Matrix & Classification Report
```

---

## 📈 Model Used

### Multinomial Naive Bayes

Multinomial Naive Bayes is widely used for text classification tasks such as:

- Spam Email Detection
- Sentiment Analysis
- Document Classification
- News Categorization

It performs efficiently on high-dimensional word-frequency data.

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## ▶️ How to Run

### Clone Repository

```bash
git clone https://github.com/Mohitjain9654/Spam-Email-Classifier.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Open the Notebook

```bash
jupyter notebook
```

or upload the notebook to **Kaggle** and run all cells.

---

## 📸 Output

The notebook displays:

- Dataset Preview
- Classification Report
- Model Accuracy
- Confusion Matrix Heatmap

---

## 💡 Future Improvements

- Compare multiple ML models (Logistic Regression, SVM, Random Forest)
- Perform feature selection
- Hyperparameter tuning
- Build a Streamlit web application
- Train using raw email text with NLP techniques such as TF-IDF

---

## 👨‍💻 Author

**Mohit Jain**

GitHub: **https://github.com/Mohitjain9654**

---

## ⭐ If you found this project useful, consider giving it a star!
