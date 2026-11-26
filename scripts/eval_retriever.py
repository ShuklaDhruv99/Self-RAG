from selfrag.retriever import get_retriever
import json
from pathlib import Path

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "eval" / "seed_questions.json"

def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding='utf-8'))
    retriever = get_retriever(k=3)

    hits = 0
    total = 0
    for q in questions:
        if not q['answerable']:
            continue
        docs = retriever.invoke(q['question'])
        pages = [d.metadata['page'] for d in docs]
        hit = q['page'] in pages
        hits += hit
        total += 1
        print(f"Q{q['id']:<2} {'HIT ' if hit else 'MISS'} expected p{q['page']}  retrieved {pages}  {q['question']}")

    print(f"\nHit rate (top 3): {hits}/{total}")

if __name__ == '__main__':
    main()
    