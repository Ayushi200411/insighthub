import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from google import genai
from dotenv import load_dotenv
from research_agent import run_research_agent
from summarizer_agent import run_summarizer_agent

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def classify_query(query):
    prompt = f"""Classify this user query into exactly ONE category. Reply with ONLY the category word, nothing else.

Categories:
- RESEARCH: questions that need answers looked up from documents (facts, explanations, details from papers)
- SUMMARIZE: requests to summarize or condense a piece of text the user provides
- DATA: requests to analyze a dataset, CSV, or numbers (not yet implemented)

Query: {query}

Category:"""

    response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    category = response.text.strip().upper()
    return category

def route(query):
    category = classify_query(query)
    print(f"[Router] Classified as: {category}")

    if category == "RESEARCH":
        return run_research_agent(query)
    elif category == "SUMMARIZE":
        return run_summarizer_agent(query)
    elif category == "DATA":
        return {"answer": "Data agent not implemented yet — coming in Week 3."}
    else:
        return {"answer": f"Could not classify query (got: {category}). Defaulting to research agent.", **run_research_agent(query)}

def main():
    print("InsightHub Agent Router — ask anything (or type 'exit')\n")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        result = route(query)
        print(f"\nAnswer:\n{result['answer']}\n")
        if "sources" in result:
            print(f"Sources: {result['sources']}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()