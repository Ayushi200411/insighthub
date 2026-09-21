import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from agents.research_agent import run_research_agent

def load_qa_pairs():
    with open("eval/qa_pairs.json", "r") as f:
        return json.load(f)

def save_results(results):
    with open("eval/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

def run_evaluation():
    qa_pairs = load_qa_pairs()
    results = []

    for i, pair in enumerate(qa_pairs):
        print(f"\n[{i+1}/{len(qa_pairs)}] {pair['question']}")

        result = None
        for attempt in range(3):
            try:
                result = run_research_agent(pair["question"])
                break
            except Exception as e:
                print(f"  API error, retrying in 10s... ({e})")
                time.sleep(10)

        if result is None:
            print("  Failed after 3 attempts, skipping this question.")
            results.append({
                "question": pair["question"],
                "expected": pair["expected_answer"],
                "system_answer": "ERROR - API unavailable",
                "sources": [],
                "grade": "skipped"
            })
            save_results(results)
            continue

        print(f"System answer: {result['answer']}")
        print(f"Expected: {pair['expected_answer']}")

        grade = input("Grade (correct/partial/wrong): ").strip().lower()
        results.append({
            "question": pair["question"],
            "expected": pair["expected_answer"],
            "system_answer": result["answer"],
            "sources": result["sources"],
            "grade": grade
        })
        save_results(results)  # save after EVERY question now

    total = len(results)
    correct = sum(1 for r in results if r["grade"] == "correct")
    partial = sum(1 for r in results if r["grade"] == "partial")
    wrong = sum(1 for r in results if r["grade"] == "wrong")

    print(f"\n{'='*50}")
    print(f"RESULTS: {correct}/{total} correct, {partial}/{total} partial, {wrong}/{total} wrong")
    print(f"Accuracy (correct only): {correct/total*100:.1f}%")
    print(f"Accuracy (correct + partial): {(correct+partial)/total*100:.1f}%")

if __name__ == "__main__":
    run_evaluation()