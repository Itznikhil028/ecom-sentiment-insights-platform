import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Config
st.set_page_config(page_title="E-com Sentiment & Rating Insights", page_icon="📊", layout="wide")

st.title("📊 E-commerce Sentiment & Rating Insights Platform")
st.write("Analyze review sentiment, predict 1-5 star ratings, and find similar dataset reviews in real time.")

# 2. Pipeline Initialization
@st.cache_resource
def load_full_pipeline():
    df = pd.read_csv("data/reviews.csv")
    df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    df['Cleaned_Review'] = df['Review_Text'].apply(clean_text)
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df['Cleaned_Review'])
    
    # Sentiment Model (Naive Bayes)
    sentiment_model = MultinomialNB()
    sentiment_model.fit(X_tfidf, df['Sentiment'])
    
    # Rating Model (Logistic Regression 1-5 Stars)
    rating_model = LogisticRegression(max_iter=1000)
    rating_model.fit(X_tfidf, df['Rating'])
    
    return df, vectorizer, X_tfidf, sentiment_model, rating_model, clean_text

df, vectorizer, X_tfidf, sentiment_model, rating_model, clean_text_func = load_full_pipeline()

# 3. Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✍️ Enter Product Review")
    user_review = st.text_area("Review Text:", placeholder="e.g., Absolutely loved the product quality and packaging!", height=150)
    analyze_btn = st.button("🚀 Run Full Analysis", use_container_width=True)

with col2:
    st.subheader("📈 AI Prediction Results")
    if analyze_btn:
        if user_review.strip() == "":
            st.warning("Please enter some text to analyze!")
        else:
            cleaned = clean_text_func(user_review)
            vec = vectorizer.transform([cleaned])
            
            # Predict Sentiment
            prob = sentiment_model.predict_proba(vec)[0]
            pos_prob = prob[1]
            
            # Predict Star Rating
            predicted_rating = rating_model.predict(vec)[0]
            
            # Sentiment Banner
            if predicted_rating >= 4:
                st.success(f"😊 **Positive Sentiment Detected!** (Confidence: {max(pos_prob, 1 - pos_prob) * 100:.2f}%)")
            else:
                st.error(f"😡 **Negative Sentiment Detected!** (Confidence: {max(pos_prob, 1 - pos_prob) * 100:.2f}%)")
            
            # Rating Metric
            st.metric(label="⭐ Predicted Rating", value=f"{predicted_rating} / 5 Stars")
            
            st.markdown("---")
            st.subheader("🔍 Similar Dataset Reviews")
            
            # Cosine Similarity for Similar Reviews
            similarities = cosine_similarity(vec, X_tfidf).flatten()
            top_indices = similarities.argsort()[-3:][::-1] # Top 3 matches
            
            for rank, idx in enumerate(top_indices, 1):
                match_score = similarities[idx] * 100
                review_text = df.iloc[idx]['Review_Text']
                rating = df.iloc[idx]['Rating']
                st.write(f"**#{rank} Match ({match_score:.1f}% Similarity) | Rating: {rating}⭐**")
                st.info(f'"{review_text}"')

st.markdown("---")
st.caption("E-com Sentiment Insights Platform • End-to-End NLP Architecture")