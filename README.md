# Zomato Reviews Sentiment Classification

## 📌 Problem Statement
This project focuses on classifying Zomato customer reviews into sentiment categories using classical Natural Language Processing (NLP) techniques and machine learning models.

The objective is to evaluate how well traditional models handle **real-world, highly imbalanced review data** and to analyze the trade-offs involved in improving minority-class performance.

---

## 🧾 Dataset Overview

Dataset from Kaggle : https://www.kaggle.com/datasets/sonalshinde123/zomato-app-play-store-reviews/data

Each review record contains:
- `review_id` – Unique identifier
- `rating` – User rating (1–5)
- `review_text` – Raw customer review
- `review_date` – Timestamp
- `helpful` – Helpful votes count (not used as target)

### Target Variable
Reviews are classified into **three sentiment classes**:
- `0` – Negative
- `1` – Neutral
- `2` – Positive

⚠️ **Class Imbalance Warning**  
The Neutral class is severely underrepresented compared to Positive and Negative reviews, making this a challenging classification task.

---

## ⚙️ Text Preprocessing
- Lowercasing
- Punctuation and noise removal
- Stopword removal
- Token normalization

### Vectorization
TF-IDF Vectorizer with:
- `max_features = 20000`
- `ngram_range = (1, 2)`
- Sparse representation for efficiency

---

## 🧠 Models Used
### 1. Logistic Regression
- Linear classifier suited for high-dimensional sparse text data
- Trained **with `class_weight="balanced"`** to address label imbalance

### 2. Multinomial Naive Bayes
- Probabilistic baseline model for text classification
- Used primarily for performance comparison

---

## 📊 Evaluation Results

### Logistic Regression (with class weighting)

* Accuracy: 0.8625
* Weighted F1 Score: 0.87
* Macro F1 Score: 0.65

| Class | Precision | Recall | F1-Score | Support |
|-----|----------|--------|---------|--------|
| 0 (Negative) | 0.82 | 0.88 | 0.85 | 1095 |
| 1 (Neutral)  | 0.13 | 0.26 | 0.18 | 133 |
| 2 (Positive) | 0.95 | 0.89 | 0.92 | 2772 |

---

### Multinomial Naive Bayes (baseline)
* Accuracy: 0.9017
* Weighted F1 Score: 0.88
* Macro F1 Score: 0.60


---

## 🔍 Key Observations
- Applying class weighting **reduced overall accuracy** but **improved minority-class recall**
- Neutral sentiment detection remains difficult due to:
  - Ambiguous language
  - Overlap with Positive/Negative classes
  - Limited training examples
- Macro F1 is a more reliable metric than accuracy for this dataset
- Improvements are constrained by **data quality**, not model choice

---

## 🚨 Trade-off Analysis
| Metric | Without Class Weight | With Class Weight |
|------|---------------------|------------------|
| Accuracy | ~90% | 86% |
| Neutral Recall | 0.00 | **0.26** |
| Macro F1 | ~0.60 | **0.65** |

This highlights the classic **fairness vs accuracy trade-off** in imbalanced classification problems.

---

## 🔧 Future Improvements
- Reformulate as binary sentiment classification (Positive vs Non-Positive)
- Merge Neutral class into ordinal sentiment buckets
- Add non-textual features:
  - Review length
  - Sentiment polarity scores
  - Rating-text mismatch indicators
- Threshold tuning for minority classes
- Evaluate using PR-AUC instead of accuracy

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorization

---

## ✅ Conclusion
This project demonstrates that classical NLP models can achieve strong performance on majority sentiment classes but struggle with minority classes in imbalanced datasets. Applying class weighting improves fairness at the cost of accuracy, reinforcing the importance of proper problem framing and evaluation metrics in real-world machine learning tasks.


