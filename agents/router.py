import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from dotenv import load_dotenv
from research_agent import run_research_agent
from summarizer_agent import run_summarizer_agent
from utils.logger import log_step
from data_agent import run_data_agent

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
    log_step("ROUTER_CLASSIFY", f"query='{query}' -> category={category}")
    return category

def route(query):
    log_step("QUERY_RECEIVED", f"query='{query}'")
    category = classify_query(query)
    print(f"[Router] Classified as: {category}")

    if category == "RESEARCH":
        log_step("DISPATCH", "-> research_agent")
        result = run_research_agent(query)
        log_step("RESEARCH_RESULT", f"sources={result.get('sources', [])}")
        return result
    elif category == "SUMMARIZE":
        log_step("DISPATCH", "-> summarizer_agent")
        result = run_summarizer_agent(query)
        log_step("SUMMARIZE_RESULT", "done")
        return result
    elif category == "DATA":
        log_step("DISPATCH", "-> data_agent")
        result = run_data_agent("data/titanic.csv", "Survived")
        log_step("DATA_RESULT", "done")
        return result
    else:
        log_step("DISPATCH_FALLBACK", f"unrecognized category={category}, defaulting to research")
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