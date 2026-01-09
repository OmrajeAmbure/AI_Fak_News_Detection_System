from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_check(news_text):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
You are a professional fact-checking AI.

Task:
1. Decide whether the following news is REAL or FAKE or UNCERTAIN
2. Give a short factual reason

News:
\"\"\"{news_text}\"\"\"

Respond strictly in this format:
VERDICT: REAL or FAKE or UNCERTAIN
REASON: one short paragraph
"""
        )

        output = response.text.strip()

        verdict = "UNCERTAIN"
        if "VERDICT: REAL" in output:
            verdict = "REAL"
        elif "VERDICT: FAKE" in output:
            verdict = "FAKE"

        return {
            "verdict": verdict,
            "reason": output
        }

    except Exception as e:
        return {
            "verdict": "ERROR",
            "reason": f"Gemini API failed: {str(e)}"
        }
