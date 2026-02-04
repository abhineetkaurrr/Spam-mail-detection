import streamlit as st
import pickle
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Spam Mail Detection",
    page_icon="📩",
    layout="centered"
)

# ---------- LOAD MODEL (CACHED) ----------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center;'>📩 Spam Mail Detection</h1>
    <p style='text-align: center; color: gray;'>
    Detect whether a message is <b>SPAM</b> or <b>HAM</b> using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------- INPUT ----------
message = st.text_area(
    "✉️ Enter the message text below:",
    height=150,
    placeholder="Win ₹10,000 cash prize! Click the link now..."
)

# ---------- BUTTON ----------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

# ---------- PREDICTION ----------
if predict_btn:
    if message.strip() == "":
        st.warning("⚠️ Please enter a message first")
    else:
        vectorized_text = vectorizer.transform([message])
        prediction = model.predict(vectorized_text)[0]

        # Probability (Naive Bayes supports this)
        try:
            probability = model.predict_proba(vectorized_text)[0]
            spam_prob = probability[1] * 100
        except:
            spam_prob = None

        st.markdown("---")

        if prediction == 1:
            st.error("🚨 **SPAM MESSAGE DETECTED**")
            if spam_prob is not None:
                st.metric("Spam Probability", f"{spam_prob:.2f}%")
        else:
            st.success("✅ **THIS MESSAGE IS NOT SPAM (HAM)**")
            if spam_prob is not None:
                st.metric("Ham Confidence", f"{100 - spam_prob:.2f}%")

# ---------- SIDEBAR ----------
st.sidebar.title("📌 About Project")
st.sidebar.info(
    """
    **Spam Mail Detection App**
    
    - Machine Learning based classifier (Naive Bayes)
    - Text vectorization using **TF-IDF**
    - Built with **Streamlit + Python**
    """
)

st.sidebar.title("⚙️ Model Info")
st.sidebar.write("✔ Trained on labeled SMS/Email dataset")
st.sidebar.write("✔ Feature extraction using TF-IDF Vectorizer")

st.sidebar.markdown("---")
st.sidebar.caption("Made by SIC-25-26-Team312 using Streamlit")
