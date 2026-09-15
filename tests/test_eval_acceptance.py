import json

from evaluation.scripts.eval_acceptance import (
    DEFAULT_QUESTIONS,
    evaluate_response,
    load_questions,
    render_report,
    summarize,
)


def test_acceptance_question_file_is_valid_and_representative():
    from pathlib import Path

    questions = load_questions(
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "data"
        / "eval_acceptance_questions.example.json"
    )

    assert len(questions) >= 4
    assert {item["language"] for item in questions} == {"en", "zh", "mixed"}
    assert {item["category"] for item in questions} >= {
        "direct",
        "paraphrase",
        "refusal",
    }
    assert any(not item["should_answer"] for item in questions)


def test_answer_evaluation_uses_keywords_and_real_source_name():
    item = {
        "id": "q1",
        "category": "direct",
        "language": "en",
        "question": "How long?",
        "expected_keywords": ["16", "weeks"],
        "expected_sources": ["letter.pdf"],
        "should_answer": True,
    }
    payload = {
        "answer": "The internship lasts 16 weeks. [letter.pdf p.1]",
        "sources": [
            {"source": "letter.pdf", "page": 1, "text": "...", "distance": 0.2}
        ],
    }

    result = evaluate_response(item, payload, 125.5)

    assert result["passed"] is True
    assert result["keyword_hits"] == {"16": True, "weeks": True}
    assert result["source_hits"] == {"letter.pdf": True}


def test_refusal_is_scored_separately_from_returned_diagnostic_sources():
    item = {
        "id": "q2",
        "category": "refusal",
        "language": "en",
        "question": "What is the Wi-Fi password?",
        "expected_keywords": [],
        "expected_sources": [],
        "should_answer": False,
    }

    passed = evaluate_response(
        item,
        {
            "answer": "I couldn't find the answer to this question in the available documents.",
            "sources": [
                {"source": "unrelated.pdf", "page": 1, "text": "...", "distance": 0.8}
            ],
        },
        50,
    )
    failed = evaluate_response(
        item,
        {"answer": "The password is 1234.", "sources": []},
        50,
    )

    assert passed["passed"] is True
    assert passed["refusal_check_passed"] is True
    assert passed["unexpected_sources_returned"] is True
    assert failed["passed"] is False


def test_summary_and_report_put_failures_first():
    results = [
        {
            "id": "pass",
            "question": "pass?",
            "should_answer": True,
            "answer": "ok",
            "latency_ms": 100.0,
            "answer_check_passed": True,
            "source_check_passed": True,
            "refusal_check_passed": None,
            "passed": True,
            "error": None,
        },
        {
            "id": "fail",
            "question": "fail?",
            "should_answer": False,
            "answer": "invented",
            "latency_ms": 300.0,
            "answer_check_passed": False,
            "source_check_passed": True,
            "refusal_check_passed": False,
            "passed": False,
            "error": None,
        },
    ]
    summary = summarize(results)
    report = render_report(
        {
            "generated_at": "2026-09-02T00:00:00+00:00",
            "base_url": "http://127.0.0.1:8000",
            "summary": summary,
            "results": results,
        }
    )

    assert summary["passed"] == 1
    assert summary["average_latency_ms"] == 200.0
    assert report.index("## Failures") < report.index("## Summary")
    assert "### fail — fail?" in report


def test_question_file_is_json_serializable():
    questions = load_questions(DEFAULT_QUESTIONS)
    assert json.loads(json.dumps(questions, ensure_ascii=False)) == questions
