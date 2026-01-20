import streamlit as st
import pickle
import string
import nltk
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- DOWNLOAD NLTK ----------------
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.fade-in {
    animation: fadeIn 1.5s ease-in;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.spam {
    background-color: #ffcccc;
    color: #b30000;
}
.not-spam {
    background-color: #ccffcc;
    color: #006600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TEXT PROCESSING ----------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words("english")]
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# ---------------- LOAD MODEL ----------------
tfidf = pickle.load(open('vect.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ---------------- ANIMATED TITLE ----------------
title_text = "📩 Email / SMS Spam Classifier"
title_placeholder = st.empty()

for i in range(len(title_text) + 1):
    title_placeholder.markdown(
        f"<h1 class='fade-in'>{title_text[:i]}</h1>",
        unsafe_allow_html=True
    )
    time.sleep(0.04)

st.markdown(
    "<p class='fade-in'>Detect whether a message is Spam or Not using Machine Learning</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- INPUT ----------------
st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
input_sms = st.text_area("✉️ Enter your message", height=120)
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ---------------- PREDICT ----------------
if st.button('🔮 Predict'):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.write("⏳ Analyzing message...")

    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.markdown(
            "<div class='result-box spam fade-in'>🚨 SPAM MESSAGE</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-box not-spam fade-in'>✅ NOT SPAM</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
