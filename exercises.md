# Day 14 — Exercises

## Part 1 — Warm-up

| Metric | Acceptable low-score scenario | Critical low-score scenario | Action |
|---|---|---|---|
| Faithfulness | A clearly labelled out-of-scope refusal has little lexical overlap. | A policy answer makes unsupported claims. | Block release; inspect evidence and add grounding guardrail. |
| Answer relevance | A broad question receives a clarification question. | The response addresses a different service. | Fix routing/prompt and add intent cases. |
| Context recall | A non-essential detail is absent. | A required policy condition is missing. | Improve query, chunking, or top-k. |
| Context precision | Benign extra chunks occur after key evidence. | Noise ranks above only relevant evidence. | Rerank and tune retriever. |
| Completeness | An optional example is omitted. | A deadline, fee, approval, or safety action is omitted. | Require coverage checklist. |

Position-bias test: score equivalent answer pairs twice with reversed order and randomised labels; a systematic first-answer advantage indicates bias. Reduce verbosity bias by scoring required facts, not length, and calibrate judges against blind human labels.

| CI metric | Deployment threshold | Reason |
|---|---:|---|
| Faithfulness | 0.80 | Unsupported policy claims can harm students. |
| Relevance | 0.70 | Response must address the requested service. |
| Completeness | 0.75 | Required conditions/actions must not be omitted. |

Run offline evaluation for every model/prompt/retrieval change, online evaluation for drift, and human review for high-impact, privacy, appeal, or exception cases.

## Part 2 — Core coding

Completed in `template.py` and `solution/solution.py`: data models, five metrics, LLM judge, benchmark runner, regression, failure analysis, and lexical reranker. Required test suite: 42/42 passed.

## Part 3 — Golden dataset & benchmark

### Exercise 3.1 — Dataset

| Item | Result |
|---|---|
| Records | 20 / 20 |
| Easy / Medium / Hard / Adversarial | 5 / 7 / 5 / 3 |
| Sources used | 10 / 10 |
| Validator | PASS |

Representative cases: E03 is direct tuition lookup; M02 combines calendar and refund rules; H04 applies the effective-date policy rule. Evidence is verbatim source text and expected answers are concise, source-grounded English.

### Exercise 3.2 — Real benchmark run

Run: 2026-08-12, `gemini-3.1-flash-lite`, `top_k=5`. Sources: `artifacts/actual_answers.json` and `artifacts/benchmark_results.json`.

| ID | Recall | Precision | Faith. | Relev. | Complete. | Overall | Pass | Failure |
|---|---:|---:|---:|---:|---:|---:|---|---|
| E01 | 1.000 | 1.000 | 1.000 | 0.571 | 1.000 | 0.857 | Yes | - |
| E02 | 1.000 | 1.000 | 0.889 | 0.857 | 1.000 | 0.915 | Yes | - |
| E03 | 1.000 | 1.000 | 1.000 | 0.778 | 1.000 | 0.926 | Yes | - |
| E04 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 0.917 | Yes | - |
| E05 | 1.000 | 0.950 | 0.517 | 0.727 | 1.000 | 0.748 | Yes | - |
| M01 | 0.913 | 1.000 | 0.760 | 0.625 | 0.739 | 0.708 | Yes | - |
| M02 | 0.789 | 0.950 | 0.059 | 0.417 | 0.211 | 0.229 | No | hallucination |
| M03 | 1.000 | 1.000 | 0.700 | 0.900 | 0.929 | 0.843 | Yes | - |
| M04 | 0.938 | 1.000 | 0.900 | 0.700 | 0.500 | 0.700 | Yes | - |
| M05 | 0.862 | 1.000 | 0.735 | 0.625 | 0.897 | 0.752 | Yes | - |
| M06 | 1.000 | 1.000 | 0.769 | 0.778 | 0.667 | 0.738 | Yes | - |
| M07 | 0.941 | 0.917 | 0.469 | 0.875 | 0.882 | 0.742 | No | off_topic |
| H01 | 0.952 | 1.000 | 0.917 | 0.200 | 0.524 | 0.547 | No | irrelevant |
| H02 | 0.579 | 0.700 | 0.577 | 1.000 | 0.526 | 0.701 | Yes | - |
| H03 | 1.000 | 1.000 | 0.529 | 0.462 | 0.474 | 0.488 | No | off_topic |
| H04 | 0.875 | 1.000 | 0.250 | 0.429 | 0.667 | 0.448 | No | hallucination |
| H05 | 0.960 | 1.000 | 0.935 | 0.667 | 0.960 | 0.854 | Yes | - |
| A01 | 0.000 | 0.000 | 0.211 | 0.364 | 0.250 | 0.275 | No | hallucination |
| A02 | 0.688 | 0.917 | 0.429 | 0.556 | 0.500 | 0.495 | No | off_topic |
| A03 | 0.944 | 1.000 | 0.960 | 0.583 | 0.889 | 0.811 | Yes | - |

Aggregate: pass rate **65.0%**; Recall **0.872**; Precision **0.922**; Faithfulness **0.680**; Relevance **0.643**; Completeness **0.731**. Failure counts: hallucination 3, off_topic 3, irrelevant 1.

Lowest cases: M02 (0.229, hallucination), A01 (0.275, hallucination), H04 (0.448, hallucination). Precision is high but faithfulness is the weakest answer-side metric; M02/A01 show retrieval gaps and H04 shows generation grounding weakness despite good contexts.

### Exercise 3.3 — LLM judge rubric

Dimensions: correctness, completeness, relevance, evidence, safety/privacy.

| Score | Domain-specific criterion |
|---:|---|
| 5 | Correct, complete, grounded, safe; includes material actions/conditions. |
| 4 | Correct and safe with one minor non-material omission. |
| 3 | Partly correct but misses a material condition. |
| 2 | Major policy error, incomplete action, or weak relevance. |
| 1 | Unsupported, unsafe, privacy-violating, or unrelated. |

Edge cases: concise correct responses, safe out-of-scope refusals, and policy answers that depend on event date. Randomise answer order, score fact coverage rather than length, use multiple judges, and calibrate with blind human labels.

### Exercise 3.5 — Reranking bonus

`rerank_by_overlap()` is implemented. Recall does not change because the retrieved set is unchanged; reranking only changes rank-aware precision. It is insufficient when no chunk contains the required evidence.
