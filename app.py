import gradio as gr
import joblib, re
import numpy as np
import textstat
import nltk
import requests
from bs4 import BeautifulSoup
from newspaper import Config, Article
nltk.download("vader_lexicon")
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack

# --- IMPORT YOUR NEW GENAI UTILS ---
from gemini_utils import gemini_check 

# ================= LOAD ML ASSETS =================
try:
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
except:
    # If model is missing, create a dummy one prevents crash during UI launch
    print("⚠️ Model files not found. App will run in 'GenAI Only' mode until trained.")
    model, vectorizer = None, None

sentiment = SentimentIntensityAnalyzer()

# ================= HELPER FUNCTIONS =================
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

# ================= ROBUST SCRAPER FUNCTION =================
def extract_text_from_url(url):
    text = ""
    
    # --- METHOD 1: Newspaper3k ---
    try:
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        config = Config()
        config.browser_user_agent = user_agent
        config.request_timeout = 10
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        text = article.text
    except Exception as e:
        print(f"Method 1 failed: {e}")

    # --- METHOD 2: Direct Requests + BS4 ---
    if len(text) < 100:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://www.google.com/'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            # CHECK FOR BLOCKING BEFORE PARSING
            if response.status_code in [403, 401]:
                return "⚠️ Error: Access Denied (403). The site blocked the scraper."
            
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            
        except Exception as e:
            return f"Error: {str(e)}"

    # --- FINAL INTELLIGENT CHECK (The Fix) ---
    # Detect if we scraped a "You are blocked" page instead of news
    block_keywords = ["access denied", "security check", "please verify you are a human", "edgesuite", "akamai"]
    if any(keyword in text.lower() for keyword in block_keywords) and len(text) < 300:
        return "⚠️ Error: The website's firewall (Akamai/Cloudflare) blocked the request. Please copy-paste the text manually."

    if len(text) < 100:
        return "⚠️ Error: Could not extract text. Please use the 'Paste Text' tab."
        
    return text
# ================= MAIN ANALYSIS ENGINE =================
def check_news(news):
    if not news or len(news.strip()) < 50:
        return "⚠️ Please provide a longer article text (min 50 chars)."

    # --- 1. Traditional ML Prediction ---
    if model and vectorizer:
        cleaned = clean_text(news)
        tfidf_vec = vectorizer.transform([cleaned])
        extra = extra_features_single(cleaned)
        final_vec = hstack([tfidf_vec, extra])

        pred = model.predict(final_vec)[0]
        prob = model.predict_proba(final_vec)[0]
        ml_conf = round(max(prob) * 100, 2)
        ml_result = "REAL" if pred == 1 else "FAKE"
    else:
        ml_result = "N/A"
        ml_conf = 0

    # --- 2. Gemini Verification ---
    gemini_data = gemini_check(news) 
    gemini_verdict = gemini_data["verdict"] 
    gemini_reason = gemini_data["reason"]

    # --- 3. FINAL DECISION LOGIC (UPDATED) ---
    
    # Priority 1: If Gemini found sources, we TRUST it (Overrides ML)
    if "Verified against:" in gemini_reason:
        final_decision = f"✅ VERIFIED {gemini_verdict} (Sources Cited)"

    # Priority 2: If ML is missing, trust Gemini
    elif ml_result == "N/A":
        final_decision = gemini_verdict

    # Priority 3: If Gemini is Uncertain, fallback to ML
    elif gemini_verdict == "UNCERTAIN":
        final_decision = f"UNCERTAIN (ML says {ml_result})"
        
    # Priority 4: Agreement
    elif gemini_verdict == ml_result:
        final_decision = f"✅ VERIFIED {gemini_verdict}"
        
    # Priority 5: Hard Conflict (No sources found)
    else:
        final_decision = f"⚠️ CONFLICT (Gemini: {gemini_verdict}, ML: {ml_result})"

    # --- 4. Output Formatting ---
    result = f"""
### 🕵️ Analysis Report
**Final Decision:** {final_decision}

---
**🧠 Traditional ML Model:**
* **Prediction:** {ml_result} 
* **Confidence:** {ml_conf}%

**🤖 Gemini 2.5 AI:**
* **Verdict:** {gemini_verdict}
* **Reasoning:** {gemini_reason}
"""
    return result

def process_url(url):
    extracted_text = extract_text_from_url(url)
    if "Error" in extracted_text:
        return extracted_text, "Could not analyze due to extraction error."
    
    analysis = check_news(extracted_text)
    return extracted_text, analysis

# ================= GRADIO UI =================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
# Updated Title
    gr.Markdown("# 📰 Fake News Detector (Hybrid ML Regression Model + LLM Gemini Model)")
    
    # The Note (Using Markdown for better formatting)
    gr.Markdown(
            """
            > **⚠️ Note:** If the **Gemini 2.5 AI** verdict is `UNCERTAIN`, please rely on the 
            > **Traditional ML Model** result, click the **Analyze** button again, 
            > or provide different news content for better context.
            """
    )
    with gr.Tabs():
        with gr.TabItem("📝 Paste Text"):
            with gr.Row():
                with gr.Column():
                    t_input = gr.Textbox(lines=10, label="News Content")
                    t_btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    t_out = gr.Markdown(label="Result")
            t_btn.click(check_news, t_input, t_out)

        with gr.TabItem("🔗 Paste URL"):
            gr.Markdown("""
            > **⚠️ Note:** Some websites may block automated scraping tools.
            > If you encounter issues, please copy-paste the article text directly in the "Paste Text" tab.
            """)
            with gr.Row():
                with gr.Column():
                    u_input = gr.Textbox(label="Article URL")
                    u_btn = gr.Button("Extract & Check", variant="primary")
                    u_preview = gr.Textbox(lines=5, label="Scraped Text", interactive=False)
                with gr.Column():
                    u_out = gr.Markdown(label="Result")
            u_btn.click(process_url, u_input, [u_preview, u_out])

if __name__ == "__main__":
    demo.launch()