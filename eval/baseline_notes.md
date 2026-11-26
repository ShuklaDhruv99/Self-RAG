# Naive RAG baseline: notes (Milestone 2)

Setup: `Product_and_Pricing.pdf` (4 pages, 11 chunks), top-k retrieval, Gemini answer generation, `scripts/eval_baseline.py` over `eval/seed_questions.json` (13 questions). Answers judged by eye.

## Retrieval
- Top-3 page hit rate: 10/10 (`scripts/eval_retriever.py`).
- Caveat: with only 11 chunks, top 3 covers over a quarter of the corpus, so this is an easy test. The check compares pages, not whether the chunk contains the answer.

## Generation results

| Q | Result | Note |
|---|--------|------|
| 1, 2, 4, 5, 6, 8, 9, 10 | Correct | |
| 3 | Incomplete | Said only Growth; missed that Enterprise also includes NexaSupport (all products). Needs facts from two chunks. |
| 7 | Mostly correct | Named NexaSecure; omitted that NexaChat also has audit logs. |
| 11 | Correct | Said it doesn't know (Telegram not mentioned). |
| 12 | Debatable | Said there is no set per-user price and described custom pricing. This matches the document, so it is not a hallucination. The seed label "unanswerable" could be relabelled "no figure given". |
| 13 | Correct | Said it doesn't know (no free trial mentioned). |

Approximate score: 11/13 clean, 1 incomplete (Q3), 1 debatable (Q12).

## Findings
1. The baseline is already strong on this small document, so a Self-RAG comparison on it alone would look flat.
2. The one real failure type is multi-chunk reasoning (Q3). ISSUP / ISUSE graders should be tested against it.
3. The model did not hallucinate on the unanswerable questions, but only 3 such questions exist, which is too few to conclude anything.

## Implications for later milestones
- Milestone 5 needs 5-10 more related documents and harder questions: multi-chunk questions, near-miss distractors, and more unanswerable questions.
- Replace the by-eye judging with an automatic check (LLM-as-judge or a rubric) once the graders exist.
- Gemini returned 503 and 504 errors during these runs. Add a shared retry with backoff before building the graders and graph, which make many calls per question.
