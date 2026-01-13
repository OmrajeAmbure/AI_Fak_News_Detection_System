import gradio as gr
import joblib, re
import numpy as np
import textstat
import nltk
import requests
from bs4 import BeautifulSoup
from scipy.sparse import hstack
from nltk.sentiment import SentimentIntensityAnalyzer
from gemini_utils import gemini_check

nltk.download("vader_lexicon")

# ================= LOAD ML ASSETS =================
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
sentiment = SentimentIntensityAnalyzer()

# ================= TEXT PROCESSING =================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    return text

def extra_features_single(text):
    return np.array([[  
        len(text),
        text.count("!"),
        text.count("?"),
        textstat.flesch_reading_ease(text),
        sentiment.polarity_scores(text)["compound"]
    ]])

# ================= ARTICLE FETCH FROM LINK =================
def fetch_article_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")
        article_text = " ".join([p.get_text() for p in paragraphs])

        if len(article_text) < 300:
            return None, "❌ Could not extract enough content from link."

        return article_text, None

    except Exception as e:
        return None, f"❌ Error fetching article: {str(e)}"

# ================= MAIN CHECK FUNCTION =================
def check_news(input_type, news_text, news_link):

    if input_type == "Paste News Text":
        if not news_text.strip():
            return "❌ Please paste a news article."
        content = news_text

    else:
        if not news_link.strip():
            return "❌ Please paste a valid news link."
        content, error = fetch_article_from_url(news_link)
        if error:
            return error

    cleaned = clean_text(content)
    tfidf_vec = vectorizer.transform([cleaned])
    extra = extra_features_single(cleaned)
    final_vec = hstack([tfidf_vec, extra])

    pred = model.predict(final_vec)[0]
    prob = model.predict_proba(final_vec)[0]

    ml_conf = round(max(prob) * 100, 2)
    ml_result = "REAL" if pred == 1 else "FAKE"

    result = f"""
🧠 **ML Prediction:** {ml_result}  
📊 **ML Confidence:** {ml_conf}%  
"""

    # Gemini fallback for low confidence
    if ml_conf < 80:
        gemini = gemini_check(content)
        result += f"""
🤖 **Gemini Verdict:** {gemini['verdict']}  
📝 **Gemini Reason:** {gemini['reason']}  
"""
        if gemini["verdict"] == ml_result:
            result += f"\n✅ **Final Decision:** {ml_result}"
        else:
            result += "\n⚠️ **Final Decision:** UNCERTAIN (Human Review Recommended)"
    else:
        result += f"\n✅ **Final Decision:** {ml_result}"

    return result

# ================= GRADIO UI =================
demo = gr.Interface(
    fn=check_news,
    inputs=[
        gr.Radio(
            ["Paste News Text", "Paste News Link"],
            label="Choose Input Type",
            value="Paste News Text"
        ),
        gr.Textbox(lines=10, label="Paste News Article"),
        gr.Textbox(label="Paste News Link (URL)")
    ],
    outputs=gr.Markdown(),
    title="📰 Fake News Detector for Students",
    description="""
An AI-powered system that helps students identify fake news by analyzing
article content or links using **Machine Learning + Gemini AI**.
""",
    theme="compact"
)
demo.launch()