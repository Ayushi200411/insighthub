from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def run_summarizer_agent(text):
    prompt = f"""Summarize the following text in 3-5 concise sentences, capturing the key points:

{text}

Summary:"""

    response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    return {"answer": response.text}