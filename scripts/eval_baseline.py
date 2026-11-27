from selfrag.rag.baseline import answer, RAGResponse
import json
from pathlib import Path

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "eval" / "seed_questions.json"

def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding='utf-8'))
    for q in questions:
        
        rag_response: RAGResponse = answer(q['question'])
        print(f"Q{q['id']:<2} {q['question']} \n expected: '{q['expected_answer']}' \n retrieved: '{rag_response.answer}'")

if __name__ == '__main__':
    main()