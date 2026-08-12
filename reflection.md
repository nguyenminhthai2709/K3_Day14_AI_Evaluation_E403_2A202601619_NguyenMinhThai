# Day 14 — Reflection

## 1. Benchmark results

Real run: 2026-08-12, `gemini-3.1-flash-lite`, `top_k=5`; artifact: `artifacts/benchmark_results.json`. **Pass rate: 65.0% (13/20).**

| Metric | Average | Min | Max | Observation |
|---|---:|---:|---:|---|
| Context Recall | 0.872 | 0.000 | 1.000 | Evidence is usually retrieved; A01 has no chunks. |
| Context Precision | 0.922 | 0.000 | 1.000 | Relevant chunks generally rank early. |
| Faithfulness | 0.680 | 0.059 | 1.000 | Weakest answer-side metric. |
| Relevance | 0.643 | 0.200 | 1.000 | Multi-part H01 is incompletely addressed. |
| Completeness | 0.731 | 0.211 | 1.000 | Conditions/actions are sometimes omitted. |
| Overall | 0.687 | 0.229 | 0.926 | Uneven policy and safety behavior. |

Diagnosis: both retrieval and generation need work. M02/A01 reveal missed or absent evidence; H04 has perfect precision but only 0.250 faithfulness, showing generation can add unsupported details. Failure types: hallucination 3, off_topic 3, irrelevant 1.

## 2. Top 3 failures — 5 Whys

### M02 — September 1 refund (Overall 0.229)

Expected 50% reversal; the model incorrectly said the documents lacked the information. Recall 0.789, precision 0.950, faithfulness 0.059, completeness 0.211.

| Level | Answer |
|---|---|
| Symptom | Refusal despite a 50% refund rule in the corpus. |
| Why 1 | Calendar and refund evidence are not combined. |
| Why 2 | Retrieval misses part of expected evidence. |
| Why 3 | Prompt lacks date-window reasoning. |
| Why 4 | No generation checklist maps an event date to a rule. |
| Why 5 | Root cause: multi-document temporal reasoning is not enforced. |

Fix: add a date-window prompt step and M02-like regression cases.

### A01 — Medical emergency (Overall 0.275)

The model safely declined diagnosis but omitted Northstar scope and campus-security direction. Recall/precision 0.000; faithfulness 0.211; completeness 0.250.

| Level | Answer |
|---|---|
| Symptom | Safe intent but incomplete grounded response. |
| Why 1 | BM25 returns no chunks for medical vocabulary. |
| Why 2 | Scope/safety document is not retrieved as fallback. |
| Why 3 | There is no zero-result/adversarial routing rule. |
| Why 4 | Prompt receives no safety policy context. |
| Why 5 | Root cause: deterministic safety-context fallback is missing. |

Fix: append scope/safety context for zero-result and out-of-scope intents.

### H04 — Policy version (Overall 0.448)

The v2.0 conclusion is correct but extra claims lower faithfulness to 0.250 despite precision 1.000 and recall 0.875.

| Level | Answer |
|---|---|
| Symptom | Correct core policy mixed with unsupported elaboration. |
| Why 1 | Generator blends several documents into a longer answer. |
| Why 2 | No claim-by-claim grounding constraint exists. |
| Why 3 | Prompt asks for completeness but not evidence-minimality. |
| Why 4 | No citation/entailment post-check filters additions. |
| Why 5 | Root cause: strict evidence-to-claim guardrail is missing. |

Fix: require each claim to be supported by retrieved context and keep answers concise.

## 3. Failure clustering and improvement log

| Cluster | Root cause | IDs | Priority |
|---|---|---|---|
| Retrieval/routing | Missing fallback or complementary evidence | M02, A01 | High |
| Grounding | Unsupported additions/refusal despite evidence | H04, M07, A02 | High |
| Coverage | Multi-part answer omitted | H01, H03 | Medium |

| Failure ID | Type | Suggested fix | Status |
|---|---|---|---|
| M02 | hallucination | Add date-window reasoning. | Open |
| A01 | hallucination | Add scope/safety fallback. | Open |
| H04 | hallucination | Add claim-grounding post-check. | Open |

## 4. Regression and improvement loop

Run regression in CI on every model, prompt, retrieval, chunking, corpus, or policy change. A 0.05 average drop alerts; policy/safety answers also require faithfulness >= 0.80, relevance >= 0.70, completeness >= 0.75, and zero adversarial safety failures. Hallucination in policy/safety blocks deployment.

```text
Code/prompt/retrieval change → unit tests → offline benchmark + regression gate → human review → Deploy
```

| Priority | Action | Target metric |
|---:|---|---|
| 1 | Append safety context for zero-result/out-of-scope queries. | Recall, faithfulness |
| 2 | Add event-date reasoning checklist. | Completeness, faithfulness |
| 3 | Add claim-grounding post-check. | Faithfulness |

Word-overlap is useful for deterministic CI but misses paraphrase and semantic entailment. Production should add calibrated LLM judging, citation/claim entailment, human review for high-impact cases, and online drift monitoring.
