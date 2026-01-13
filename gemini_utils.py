from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os
import re
import datetime

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_check(news_text):
    MODEL_NAME = "gemini-2.5-flash"
    
    # 1. Get Today's Date
    today = datetime.date.today().strftime("%B %d, %Y")
    
    try:
        # 2. Enable Google Search Tool (The "Grounding" Fix)
        search_tool = Tool(google_search=GoogleSearch())
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"""
Today's Date: {today}
You are a professional fact-checker with access to Google Search.

Task:
1. Google the key entities (names, events, dates) in the text below.
2. Verify if these events actually happened recently.
3. If confirmed by search results, mark as REAL even if they seem new.

News:
\"\"\"{news_text[:4000]}\"\"\"  

Respond ONLY in this format:
VERDICT: REAL / FAKE / UNCERTAIN
REASON: [Cite the source found via Google Search]
""",
            config=GenerateContentConfig(
                tools=[search_tool],
                response_modalities=["TEXT"]
            )
        )

        output = response.text.strip()

        # 3. Robust Parsing
        verdict_match = re.search(r"VERDICT:\s*(REAL|FAKE|UNCERTAIN)", output, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.*)", output, re.IGNORECASE | re.DOTALL)

        verdict = verdict_match.group(1).upper() if verdict_match else "UNCERTAIN"
        reason = reason_match.group(1).strip() if reason_match else "No explanation provided."
        
        # 4. Extract Search Sources (Optional Bonus for your UI)
        # If the API returns web sources, you can append them to the reason
        if response.candidates[0].grounding_metadata.grounding_chunks:
            sources = [c.web.title for c in response.candidates[0].grounding_metadata.grounding_chunks]
            reason += f"\n(Verified against: {', '.join(sources[:3])})"

        return {"verdict": verdict, "reason": reason}

    except Exception as e:
        return {"verdict": "UNCERTAIN", "reason": f"Error: {str(e)}"}