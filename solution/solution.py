"""Completed evaluation core for Day 14."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or", "it", "its", "this", "that", "these", "those", "from", "into", "than"}


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"\b\w+\b", (text or "").lower()) if token not in STOPWORDS}


def _overlap(numerator: set[str], denominator: set[str]) -> float:
    if not denominator:
        return 1.0
    return max(0.0, min(1.0, len(numerator & denominator) / len(denominator)))


@dataclass
class QAPair:
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)


@dataclass
class EvalResult:
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


class RAGASEvaluator:
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        return _overlap(_tokenize(answer), _tokenize(answer)) if not _tokenize(answer) else _overlap(_tokenize(context), _tokenize(answer))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        return _overlap(_tokenize(answer), _tokenize(question))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        return _overlap(_tokenize(answer), _tokenize(expected))

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        union: set[str] = set()
        for chunk in contexts:
            union.update(_tokenize(chunk))
        return _overlap(union, _tokenize(expected))

    def evaluate_context_precision(self, contexts: list[str], expected: str, relevance_threshold: float = 0.1) -> float:
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        relevant = [len(_tokenize(chunk) & expected_tokens) / len(expected_tokens) >= relevance_threshold for chunk in contexts]
        total_relevant = sum(relevant)
        if not total_relevant:
            return 0.0
        hits = 0
        average_precision = 0.0
        for rank, is_relevant in enumerate(relevant, 1):
            if is_relevant:
                hits += 1
                average_precision += hits / rank
        return average_precision / total_relevant

    def run_full_eval(self, answer: str, question: str, context: str, expected: str, contexts: list[str] | None = None) -> EvalResult:
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)
        passed = all(score >= 0.5 for score in (faithfulness, relevance, completeness))
        failure_type = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"
        pair = QAPair(question, expected, context, retrieved_contexts=list(contexts or []))
        return EvalResult(pair, answer, faithfulness, relevance, completeness, passed, failure_type,
                          self.evaluate_context_precision(contexts, expected) if contexts is not None else None,
                          self.evaluate_context_recall(contexts, expected) if contexts is not None else None)


def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    query_tokens = _tokenize(query)
    return sorted(contexts, key=lambda chunk: len(_tokenize(chunk) & query_tokens), reverse=True)


class LLMJudge:
    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(self, question: str, answer: str, rubric: dict[str, Any]) -> dict[str, Any]:
        prompt = f"Judge the answer against this rubric. Return a JSON object mapping each criterion to 0-1.\nQuestion: {question}\nAnswer: {answer}\nRubric: {json.dumps(rubric)}"
        reasoning = self.judge_llm_fn(prompt)
        defaults = {criterion: 0.5 for criterion in rubric}
        try:
            parsed = json.loads(reasoning)
            raw_scores = parsed.get("scores", parsed) if isinstance(parsed, dict) else {}
            scores = {criterion: max(0.0, min(1.0, float(raw_scores[criterion]))) if criterion in raw_scores else 0.5 for criterion in rubric}
        except (json.JSONDecodeError, TypeError, ValueError):
            scores = defaults
        return {"scores": scores, "reasoning": reasoning}

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        means = [sum(item.get("scores", {}).values()) / len(item.get("scores", {})) for item in scores_batch if item.get("scores")]
        average = sum(means) / len(means) if means else 0.0
        positional = len(means) > 1 and means[0] > sum(means[1:]) / len(means[1:]) + 0.1
        return {"positional_bias": positional, "leniency_bias": average > 0.8, "severity_bias": average < 0.3}


class BenchmarkRunner:
    def run(self, qa_pairs: list[QAPair], agent_fn: Callable[[str], str], evaluator: RAGASEvaluator) -> list[EvalResult]:
        results = []
        for pair in qa_pairs:
            result = evaluator.run_full_eval(agent_fn(pair.question), pair.question, pair.context, pair.expected_answer, pair.retrieved_contexts)
            result.qa_pair = pair
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        def average(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None
        total = len(results)
        failure_types: dict[str, int] = {}
        for result in results:
            if result.failure_type:
                failure_types[result.failure_type] = failure_types.get(result.failure_type, 0) + 1
        return {"total": total, "passed": sum(r.passed for r in results), "pass_rate": sum(r.passed for r in results) / total if total else 0.0,
                "avg_faithfulness": average([r.faithfulness for r in results]) or 0.0,
                "avg_relevance": average([r.relevance for r in results]) or 0.0,
                "avg_completeness": average([r.completeness for r in results]) or 0.0,
                "avg_context_recall": average([r.context_recall for r in results if r.context_recall is not None]),
                "avg_context_precision": average([r.context_precision for r in results if r.context_precision is not None]),
                "failure_types": failure_types}

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        def avg(items: list, metric: str) -> float:
            return sum(getattr(item, metric) for item in items) / len(items) if items else 0.0
        report = {}
        regressions = []
        for metric in ("faithfulness", "relevance", "completeness"):
            new_value, base_value = avg(new_results, metric), avg(baseline_results, metric)
            report[f"new_avg_{metric}"] = new_value
            report[f"baseline_avg_{metric}"] = base_value
            if base_value - new_value > 0.05:
                regressions.append(metric)
        report["regressions"] = regressions
        report["passed"] = not regressions
        return report

    def identify_failures(self, results: list[EvalResult], threshold: float = 0.5) -> list[EvalResult]:
        return [r for r in results if min(r.faithfulness, r.relevance, r.completeness) < threshold]


class FailureAnalyzer:
    def categorize_failures(self, failures: list[EvalResult]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for failure in failures:
            if failure.failure_type:
                counts[failure.failure_type] = counts.get(failure.failure_type, 0) + 1
        return counts

    def find_root_cause(self, failure: EvalResult) -> str:
        scores = {"faithfulness": failure.faithfulness, "relevance": failure.relevance, "completeness": failure.completeness}
        low = min(scores, key=scores.get)
        if list(scores.values()).count(scores[low]) > 1:
            return "Multiple issues detected — review full pipeline"
        return {"faithfulness": "Context is missing or irrelevant — improve retrieval", "relevance": "Answer does not address the question — improve prompt clarity", "completeness": "Answer is missing key information — increase context window or improve generation"}[low]

    def generate_improvement_suggestions(self, failures: list[EvalResult]) -> list[str]:
        if not failures:
            return []
        categories = self.categorize_failures(failures)
        options = []
        if any("hallucination" in key.lower() for key in categories): options.append("Add grounded-answer checks and retrieve evidence before generating claims.")
        if any("irrelevant" in key.lower() or "off_topic" in key.lower() for key in categories): options.append("Clarify the routing prompt and add intent-specific examples.")
        if any("incomplete" in key.lower() or "completeness" in key.lower() for key in categories): options.append("Retrieve more complementary chunks and require answers to cover each requested condition.")
        defaults = ["Add regression cases for each observed failure before deploying a fix.", "Review low-scoring traces weekly and calibrate thresholds with human labels.", "Measure retrieval recall separately from generation quality."]
        return (options + defaults)[:max(3, len(options))]

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        lines = ["| Failure ID | Type | Root Cause | Suggested Fix | Status |", "|------------|------|------------|---------------|--------|"]
        for index, failure in enumerate(failures, 1):
            failure_id = failure.qa_pair.metadata.get("id", f"F{index:03d}")
            fix = suggestions[index - 1] if index <= len(suggestions) else "Review and prioritize a targeted fix."
            lines.append(f"| {failure_id} | {failure.failure_type or 'unknown'} | {self.find_root_cause(failure)} | {fix} | Open |")
        return "\n".join(lines)
