"""通过现有 FastAPI /query 运行轻量 RAG acceptance evaluation。

此脚本只读取 /health 并调用 /query，不直接访问或修改 ChromaDB。
运行真实评测会调用已配置的外部 LLM，可能产生费用。
"""

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).parent
DEFAULT_QUESTIONS = ROOT / "eval_acceptance_questions.json"
DEFAULT_RESULTS = ROOT / "eval_acceptance_results.json"
DEFAULT_REPORT = ROOT / "eval_acceptance_report.md"

REFUSAL_MARKERS = (
    "资料中找不到",
    "知识库是空的",
    "couldn't find the answer",
    "knowledge base is empty",
    "not available in the provided documents",
    "documents do not contain",
)


def load_questions(path: Path) -> list[dict[str, Any]]:
    """读取并校验 acceptance 题集的最小 schema。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Question file must contain a non-empty JSON array.")

    required = {
        "id",
        "category",
        "language",
        "question",
        "expected_keywords",
        "expected_sources",
        "should_answer",
    }
    seen_ids: set[str] = set()
    for index, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object.")
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Question {index} is missing: {sorted(missing)}")
        if not isinstance(item["id"], str) or not item["id"].strip():
            raise ValueError(f"Question {index} has an invalid id.")
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate question id: {item['id']}")
        seen_ids.add(item["id"])
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(f"Question {item['id']} has an invalid question.")
        if not isinstance(item["expected_keywords"], list) or not all(
            isinstance(value, str) and value for value in item["expected_keywords"]
        ):
            raise ValueError(f"Question {item['id']} has invalid expected_keywords.")
        if not isinstance(item["expected_sources"], list) or not all(
            isinstance(value, str) and value for value in item["expected_sources"]
        ):
            raise ValueError(f"Question {item['id']} has invalid expected_sources.")
        if not isinstance(item["should_answer"], bool):
            raise ValueError(f"Question {item['id']} has invalid should_answer.")

    return data


def percentile(values: list[float], percentile_value: int) -> float:
    """使用 nearest-rank 计算小样本延迟分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile_value / 100 * len(ordered)))
    return ordered[rank - 1]


def is_refusal(answer: str) -> bool:
    normalized = answer.casefold()
    return any(marker.casefold() in normalized for marker in REFUSAL_MARKERS)


def evaluate_response(
    item: dict[str, Any], payload: dict[str, Any], latency_ms: float
) -> dict[str, Any]:
    """对可变自然语言答案做 keyword/source/refusal 检查。"""
    answer = payload.get("answer", "")
    sources = payload.get("sources", [])
    if not isinstance(answer, str) or not isinstance(sources, list):
        raise ValueError("/query returned an unexpected response schema.")

    answer_folded = answer.casefold()
    keyword_hits = {
        keyword: keyword.casefold() in answer_folded
        for keyword in item["expected_keywords"]
    }
    returned_source_names = {
        source.get("source") for source in sources if isinstance(source, dict)
    }
    source_hits = {
        source: source in returned_source_names
        for source in item["expected_sources"]
    }

    if item["should_answer"]:
        answer_check = bool(keyword_hits) and all(keyword_hits.values())
        source_check = bool(source_hits) and any(source_hits.values())
        refusal_check = None
        unexpected_sources_returned = False
    else:
        refusal_check = is_refusal(answer)
        answer_check = refusal_check
        source_check = True
        unexpected_sources_returned = bool(sources)

    return {
        "id": item["id"],
        "category": item["category"],
        "language": item["language"],
        "question": item["question"],
        "should_answer": item["should_answer"],
        "answer": answer,
        "sources": sources,
        "latency_ms": round(latency_ms, 2),
        "keyword_hits": keyword_hits,
        "source_hits": source_hits,
        "answer_check_passed": answer_check,
        "source_check_passed": source_check,
        "refusal_check_passed": refusal_check,
        "unexpected_sources_returned": unexpected_sources_returned,
        "passed": answer_check and source_check,
        "error": None,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [result["latency_ms"] for result in results]
    refusal_results = [result for result in results if not result["should_answer"]]
    return {
        "total_questions": len(results),
        "passed": sum(bool(result["passed"]) for result in results),
        "answer_checks_passed": sum(
            bool(result["answer_check_passed"]) for result in results
        ),
        "source_checks_passed": sum(
            bool(result["source_check_passed"]) for result in results
        ),
        "refusal_checks_passed": sum(
            result["refusal_check_passed"] is True for result in refusal_results
        ),
        "refusal_checks_total": len(refusal_results),
        "average_latency_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": round(percentile(latencies, 50), 2),
        "p95_latency_ms": round(percentile(latencies, 95), 2),
    }


def render_report(run: dict[str, Any]) -> str:
    summary = run["summary"]
    failures = [result for result in run["results"] if not result["passed"]]
    lines = [
        "# RAG Acceptance Report",
        "",
        f"Generated: `{run['generated_at']}`  ",
        f"API: `{run['base_url']}`",
        "",
        "## Failures",
        "",
    ]
    if not failures:
        lines.append("No acceptance failures.")
    else:
        for result in failures:
            lines.extend(
                [
                    f"### {result['id']} — {result['question']}",
                    "",
                    f"- Error: `{result['error']}`" if result["error"] else "- Error: none",
                    f"- Answer check: `{result['answer_check_passed']}`",
                    f"- Source check: `{result['source_check_passed']}`",
                    f"- Refusal check: `{result['refusal_check_passed']}`",
                    f"- Unexpected sources returned: `{result.get('unexpected_sources_returned', False)}`",
                    f"- Latency: `{result['latency_ms']} ms`",
                    f"- Answer: {result['answer'] or '(no answer)' }",
                    "",
                ]
            )

    lines.extend(
        [
            "## Summary",
            "",
            f"- Total questions: {summary['total_questions']}",
            f"- Overall passed: {summary['passed']}/{summary['total_questions']}",
            f"- Answer checks passed: {summary['answer_checks_passed']}/{summary['total_questions']}",
            f"- Citation/source checks passed: {summary['source_checks_passed']}/{summary['total_questions']}",
            f"- Refusal checks passed: {summary['refusal_checks_passed']}/{summary['refusal_checks_total']}",
            f"- Average latency: {summary['average_latency_ms']} ms",
            f"- P50 latency: {summary['p50_latency_ms']} ms",
            f"- P95 latency: {summary['p95_latency_ms']} ms",
            "",
            "## Per-question results",
            "",
            "| ID | Passed | Answer | Source | Refusal | Latency ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in run["results"]:
        lines.append(
            f"| {result['id']} | {result['passed']} | "
            f"{result['answer_check_passed']} | {result['source_check_passed']} | "
            f"{result['refusal_check_passed']} | {result['latency_ms']} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_acceptance(
    questions: list[dict[str, Any]], base_url: str, timeout_seconds: float
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    normalized_base_url = base_url.rstrip("/")

    with httpx.Client(base_url=normalized_base_url, timeout=timeout_seconds) as client:
        health = client.get("/health")
        health.raise_for_status()
        health_payload = health.json()
        if health_payload.get("chunks", 0) <= 0:
            raise RuntimeError("Knowledge base is empty; acceptance run was not started.")

        for index, item in enumerate(questions, 1):
            print(f"[{index}/{len(questions)}] {item['id']}: {item['question']}")
            started = time.perf_counter()
            try:
                response = client.post("/query", json={"question": item["question"]})
                response.raise_for_status()
                latency_ms = (time.perf_counter() - started) * 1000
                result = evaluate_response(item, response.json(), latency_ms)
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                result = {
                    "id": item["id"],
                    "category": item["category"],
                    "language": item["language"],
                    "question": item["question"],
                    "should_answer": item["should_answer"],
                    "answer": "",
                    "sources": [],
                    "latency_ms": round(latency_ms, 2),
                    "keyword_hits": {},
                    "source_hits": {},
                    "answer_check_passed": False,
                    "source_check_passed": False,
                    "refusal_check_passed": False if not item["should_answer"] else None,
                    "unexpected_sources_returned": False,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": normalized_base_url,
        "summary": summarize(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the question file without calling the API or LLM.",
    )
    parser.add_argument(
        "--rescore-existing",
        action="store_true",
        help="Re-score the existing results file without calling the API or LLM.",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.validate_only:
        print(f"Validated {len(questions)} acceptance questions: {args.questions}")
        return

    if args.rescore_existing:
        existing = json.loads(args.results.read_text(encoding="utf-8"))
        questions_by_id = {item["id"]: item for item in questions}
        rescored = []
        for old_result in existing["results"]:
            if old_result.get("error"):
                old_result.setdefault("unexpected_sources_returned", False)
                rescored.append(old_result)
                continue
            item = questions_by_id[old_result["id"]]
            rescored.append(
                evaluate_response(
                    item,
                    {
                        "answer": old_result["answer"],
                        "sources": old_result["sources"],
                    },
                    old_result["latency_ms"],
                )
            )
        run = {
            "generated_at": existing["generated_at"],
            "base_url": existing["base_url"],
            "summary": summarize(rescored),
            "results": rescored,
        }
    else:
        run = run_acceptance(questions, args.base_url, args.timeout)
    args.results.write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.write_text(render_report(run), encoding="utf-8")
    print(json.dumps(run["summary"], ensure_ascii=False, indent=2))
    print(f"Results: {args.results}")
    print(f"Report:  {args.report}")


if __name__ == "__main__":
    main()
