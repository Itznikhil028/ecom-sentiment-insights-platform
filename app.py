import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Page Configuration (UI setup)
st.set_page_config(page_title="E-com Sentiment Insights", page_icon="📊", layout="centered")

st.title("📊 E-commerce Sentiment Insights Platform")
st.write("Type a product review below to instantly predict its emotional sentiment using our trained Naive Bayes AI model.")

# --- Mock Data Pipeline Setup for Demo Purpose ---
# Kyunki humne model deploy karna hai, hum pipeline ko instantly initialize kar rahe hain
@st.cache_resource
def load_trained_pipeline():
    # Load raw dataset to replicate the exact vocabulary state
    df = pd.read_csv("data/reviews.csv")
    df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    df['Cleaned_Review'] = df['Review_Text'].apply(clean_text)
    
    # Re-train upgraded vectorizer and model for the web interface
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df['Cleaned_Review'])
    
    model = MultinomialNB()
    model.fit(X_tfidf, df['Sentiment'])
    
    return vectorizer, model, clean_text

vectorizer, model, clean_text_func = load_trained_pipeline()

# 2. User Input UI Section
user_review = st.text_area("✍️ Enter Product Review:", placeholder="e.g., The product quality is great, absolutely loved it!")

if st.button("🚀 Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter some text to analyze!")
    else:
        # Preprocessing & Inference
        cleaned = clean_text_func(user_review)
        vectorized = vectorizer.transform([cleaned])
        
        prob = model.predict_proba(vectorized)[0]
        positive_prob = prob[1]
        
        # Sentiment logic matching our Jupyter analytics state
        if positive_prob >= 0.45:
            st.success(f"😊 **Positive Sentiment Detected!** (Confidence: {positive_prob * 100:.2f}%)")
        else:
            st.error(f"😡 **Negative Sentiment Detected!** (Confidence: {(1 - positive_prob) * 100:.2f}%)")

st.markdown("---")
st.caption("Developed as part of the E-com Sentiment Insights Architecture.")