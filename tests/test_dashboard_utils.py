from dashboard.utils import failed_cases, judge_rows, review_queue, slice_accuracy
from evalcascade.models import AggregateMetric, CaseResult, EvaluationResult


def test_judge_rows_keep_rubric_score_and_review_reason():
    cases = [
        CaseResult(
            run_id="r",
            case_id="en-comp-judge-low-001",
            application_status="success",
            final_status="review",
            annotations={"judge_rubric": "Name the inverse reading."},
            evaluator_results=[
                EvaluationResult(
                    status="not_applicable",
                    evaluator_name="schema",
                    evaluator_version="1.0.0",
                ),
                EvaluationResult(
                    status="review",
                    score=1.0,
                    category="LLM_JUDGE_RESULT",
                    evidence={"confidence": 0.4, "parsed_score": 1.0},
                    evaluator_name="llm_judge",
                    evaluator_version="1.0.0",
                ),
            ],
        )
    ]
    rows = judge_rows(cases)
    assert rows == [
        {
            "case_id": "en-comp-judge-low-001",
            "rubric": "Name the inverse reading.",
            "score": 1.0,
            "confidence": 0.4,
            "status": "review",
            "review_reason": "confidence below floor",
        }
    ]


def test_review_queue_filters():
    cases = [
        CaseResult(
            run_id="r",
            case_id="a",
            application_status="success",
            final_status="fail",
            review_required=True,
        ),
        CaseResult(
            run_id="r",
            case_id="b",
            application_status="success",
            final_status="pass",
            review_required=False,
        ),
    ]
    assert [item.case_id for item in review_queue(cases)] == ["a"]
    assert [item.case_id for item in failed_cases(cases)] == ["a"]


def test_slice_accuracy_rows():
    metrics = [
        AggregateMetric(run_id="r", metric_name="overall_accuracy", value=0.9),
        AggregateMetric(
            run_id="r",
            metric_name="overall_accuracy",
            value=0.5,
            slice_name="severity",
            slice_value="high",
        ),
    ]
    rows = slice_accuracy(metrics)
    assert rows == [{"slice": "severity=high", "accuracy": 0.5}]
    metrics.extend(
        [
            AggregateMetric(
                run_id="r",
                metric_name="overall_accuracy_ci_low",
                value=0.1,
                slice_name="severity",
                slice_value="high",
            ),
            AggregateMetric(
                run_id="r",
                metric_name="overall_accuracy_ci_high",
                value=0.9,
                slice_name="severity",
                slice_value="high",
            ),
            AggregateMetric(
                run_id="r",
                metric_name="majority_class_baseline",
                value=0.5,
                slice_name="severity",
                slice_value="high",
            ),
        ]
    )
    enriched = slice_accuracy(metrics)[0]
    assert enriched["ci"] == "[10.0%, 90.0%]"
    assert enriched["majority"] == 0.5
