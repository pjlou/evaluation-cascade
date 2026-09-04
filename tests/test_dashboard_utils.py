from dashboard.utils import failed_cases, review_queue, slice_accuracy
from evalcascade.models import AggregateMetric, CaseResult


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
