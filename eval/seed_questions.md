# Seed questions: Product_and_Pricing.pdf (4 pages)

Page numbers below are **1-based, as printed in the PDF**. `load_docs()` in `loader.py`
converts `PyPDFLoader`'s 0-based page to 1-based, so `metadata["page"]` matches these directly (no `+ 1`).

| # | Question | Expected answer | Page (1-based) | Type |
|---|----------|-----------------|----------------|------|
| 1 | How much does the Growth plan cost per user per month? | 5,999 | 3 | factual |
| 2 | How many queries per month does the Starter plan allow? | Up to 5,000 | 3 | factual |
| 3 | Which plan includes NexaSupport? | Growth (and Enterprise, which includes all products) | 3 | factual |
| 4 | What discount does annual billing give? | 15% | 4 | factual (heading on p3, answer on p4) |
| 5 | Does NexaChat support role-based access control? | Yes (HR, Finance, Sales, Engineering) | 1 | factual |
| 6 | What does NexaSupport do when it cannot handle a query confidently? | Escalates to a human agent (confidence-based escalation) | 2 | factual |
| 7 | Which product provides audit trails and query logging for AI usage? | NexaSecure (NexaChat also has audit logs) | 2 | factual |
| 8 | Which plan offers on-premise or private cloud deployment, and is it priced per user? | Enterprise; custom pricing based on usage and deployment model | 3 | multi-fact |
| 9 | Which products can export reports to PDF and Excel? | NexaInsight | 1 | factual |
| 10 | What support does the Enterprise plan include? | 24x7 support with SLA, dedicated customer success manager | 3 | factual |

## Unanswerable (the document does not say; the system should admit it doesn't know)

| # | Question | Why unanswerable |
|---|----------|------------------|
| 11 | Does NexaChat integrate with Telegram? | Only Slack, CRM and BI tools are mentioned (Growth plan) |
| 12 | What is the Enterprise plan's price per user? | Pricing is "custom", no figure given |
| 13 | Does NexaAI offer a free trial? | Not mentioned anywhere |

## Notes
- Q3 and Q8 need information that may sit in different chunks; likely retrieval stress tests.
- Q4: the "Billing & Discounts" heading is at the bottom of page 3 and the content is on page 4, so a page-boundary split can hurt retrieval.
- Q6 is not in the first example set in `retriever.py`; Q1-Q5 and Q11 overlap with it.
- Hit-rate rule: a question counts as a hit if a chunk from the expected page is in the top 3. Q11-Q13 have no expected page, so skip them for the hit rate; they are for the generation step in Milestone 2.
