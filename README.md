# 📊 E-Commerce Sentiment & Rating Insights Platform

An end-to-end Machine Learning & Natural Language Processing (NLP) system designed to analyze Amazon/E-commerce product reviews. It classifies review sentiment, predicts star ratings (1–5 Stars), generates Word Clouds, and finds similar reviews in real time using Cosine Similarity.

---

## 📌 Project Overview & Features

- **Text Preprocessing Pipeline:** Automatic text cleaning using Regex (lowercasing, punctuation removal, whitespace cleaning).
- **TF-IDF Feature Extraction:** Converts text into numerical vectors using Unigrams and Bigrams.
- **Multi-Model Machine Learning:** Benchmarked using Logistic Regression, Naive Bayes, and Random Forest models.
- **Dual Inference Engine:** 
  - Binary Sentiment Prediction (Positive / Negative) with confidence score.
  - Multi-class Rating Prediction (1 to 5 Stars).
- **Visual Analytics:** Sentiment-based Word Clouds highlighting positive and negative review keywords.
- **Similarity Search:** Integrated Cosine Similarity to find top matching reviews from the dataset.
- **Interactive Web App:** Synchronized Streamlit application for real-time predictions.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib, WordCloud, Streamlit
- **Version Control:** Git & GitHub

---

## 📂 Project Structure

```text
ecom-sentiment-insights-platform/
│
├── data/
│   └── reviews.csv              # Product reviews dataset
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb # Data cleaning and EDA
│   └── 02_model_training.ipynb    # Model training & WordClouds
│
├── app.py                       # Streamlit Web Application
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation


🚀 How to Run the Project
Install dependencies:

Bash
pip install -r requirements.txt
Launch the Streamlit Web Application:

Bash
streamlit run app.py


🙏 Acknowledgments
Special thanks to mentor Adtya Gupta and MindCoders - The Future Is Here for their practical guidance in Python with AI/ML.


📝 License
This project is licensed under the MIT License.