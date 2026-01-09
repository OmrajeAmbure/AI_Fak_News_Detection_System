import gradio as gr
import joblib, re
import numpy as np
import textstat
import nltk
nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from gemini_utils import gemini_check


# Load ML assets
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
sentiment = SentimentIntensityAnalyzer()

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

def check_news(news):
    cleaned = clean_text(news)
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

    if ml_conf < 80:
        gemini = gemini_check(news)
        result += f"""
🤖 **Gemini Verdict:** {gemini['verdict']}  
📝 **Gemini Reason:** {gemini['reason']}  
"""
        if gemini["verdict"] == ml_result:
            result += f"\n✅ **Final Decision:** {ml_result}"
        else:
            result += "\n⚠️ **Final Decision:** UNCERTAIN (Human review needed)"
    else:
        result += f"\n✅ **Final Decision:** {ml_result}"

    return result


demo = gr.Interface(
    fn=check_news,
    inputs=gr.Textbox(lines=10, label="Paste News Article"),
    outputs=gr.Markdown(),
    title="📰 AI Fake News Detection System",
    description="Hybrid ML + Gemini AI based Fake News Detection System. Paste a news article to check its authenticity.",
    theme="compact"
)

demo.launch()
