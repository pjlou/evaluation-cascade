from evalcascade.cascade import Cascade, build_cascade
from evalcascade.config import RunConfig
from evalcascade.datasets import load_dataset
from evalcascade.evaluators.llm_judge import LlmJudgeEvaluator
from evalcascade.evaluators.rule_graph import RuleGraphEvaluator
from evalcascade.evaluators.schema import SchemaEvaluator
from evalcascade.models import ApplicationOutput, EvaluationCase, EvaluationResult


def _case(**kwargs) -> EvaluationCase:
    payload = {
        "id": "judge-1",
        "input": "explain the compositional reading",
        "expected": {},
        "severity": "medium",
        "metadata": {"judge_rubric": "Score 1 if the explanation is logically valid, else 0."},
        "dataset_version": "smoke-v1",
    }
    payload.update(kwargs)
    return EvaluationCase.model_validate(payload)


def _judge_fn(score: float, confidence: float):
    def fn(item, model_output, rubric):
        return {
            "judge_model": "llama3.1:8b",
            "parsed_score": score,
            "confidence": confidence,
            "rationale": "stub",
        }

    return fn


def test_not_applicable_without_rubric():
    case = _case(metadata={})
    result = LlmJudgeEvaluator(judge_fn=_judge_fn(1.0, 1.0)).evaluate(
        case, ApplicationOutput(status="success", raw_text="x", output="x"), []
    )
    assert result.status == "not_applicable"


def test_not_applicable_when_a_deterministic_evaluator_already_resolved():
    prior = EvaluationResult(status="pass", evaluator_name="schema", evaluator_version="1.0.0")
    result = LlmJudgeEvaluator(judge_fn=_judge_fn(1.0, 1.0)).evaluate(
        _case(), ApplicationOutput(status="success", raw_text="x", output="x"), [prior]
    )
    assert result.status == "not_applicable"
    assert result.evidence["resolved_by"] == ["schema"]


def test_application_error_fails_without_calling_judge():
    def boom(item, model_output, rubric):
        raise AssertionError("judge_fn should not be called on application error")

    result = LlmJudgeEvaluator(judge_fn=boom).evaluate(
        _case(), ApplicationOutput(status="timeout", error="no response"), []
    )
    assert result.status == "fail"
    assert result.category == "application_error"


def test_empty_output_routes_to_review_without_calling_judge():
    def boom(item, model_output, rubric):
        raise AssertionError("judge_fn should not be called on empty output")

    result = LlmJudgeEvaluator(judge_fn=boom).evaluate(
        _case(), ApplicationOutput(status="success", raw_text="   ", output="   "), []
    )
    assert result.status == "review"
    assert result.category == "EMPTY_OR_MALFORMED_OUTPUT"


def test_high_confidence_high_score_passes():
    result = LlmJudgeEvaluator(judge_fn=_judge_fn(0.9, 0.95)).evaluate(
        _case(), ApplicationOutput(status="success", raw_text="reasoning", output="reasoning"), []
    )
    assert result.status == "pass"
    assert result.score == 0.9


def test_high_confidence_low_score_fails():
    result = LlmJudgeEvaluator(judge_fn=_judge_fn(0.1, 0.95)).evaluate(
        _case(), ApplicationOutput(status="success", raw_text="reasoning", output="reasoning"), []
    )
    assert result.status == "fail"


def test_low_confidence_routes_to_review_even_with_high_score():
    result = LlmJudgeEvaluator(judge_fn=_judge_fn(0.9, 0.2)).evaluate(
        _case(), ApplicationOutput(status="success", raw_text="reasoning", output="reasoning"), []
    )
    assert result.status == "review"


def test_judge_invocation_error_is_captured_not_raised():
    def boom(item, model_output, rubric):
        raise ValueError("judge score out of range")

    result = LlmJudgeEvaluator(judge_fn=boom).evaluate(
        _case(), ApplicationOutput(status="success", raw_text="reasoning", output="reasoning"), []
    )
    assert result.status == "error"
    assert result.category == "JUDGE_INVOCATION_ERROR"


def test_cascade_fail_from_judge_is_not_overridden_by_review():
    from evalcascade.evaluators.review import ReviewEvaluator

    cascade = Cascade([SchemaEvaluator(), LlmJudgeEvaluator(judge_fn=_judge_fn(0.1, 0.95)), ReviewEvaluator()])
    outcome = cascade.evaluate(
        _case(severity="high"),
        ApplicationOutput(status="success", raw_text="reasoning", output="reasoning"),
    )
    assert outcome.final_status == "fail"
    assert outcome.deciding_evaluator == "llm_judge"


def test_build_cascade_recognizes_llm_judge():
    config = RunConfig(evaluators=["schema", "llm_judge", "review"])
    cascade = build_cascade(config)
    assert [item.name for item in cascade.evaluators] == ["schema", "llm_judge", "review_router"]
    judge = cascade.evaluators[1]
    assert judge._judge_fn is not None


def test_mock_judge_routes_low_confidence_fixture_to_review():
    from evalcascade.adapters.mock import MockAdapter

    cases = load_dataset("smoke-v1")
    probe = next(case for case in cases if case.id == "en-comp-judge-low-001")
    config = RunConfig(evaluators=["schema", "rule_graph", "llm_judge", "review"], adapter={"name": "mock"})
    cascade = build_cascade(config)
    output = MockAdapter().run(probe, {})
    outcome = cascade.evaluate(probe, output)
    assert outcome.final_status == "review"
    judge = next(item for item in outcome.evaluator_results if item.evaluator_name == "llm_judge")
    assert judge.status == "review"
    assert judge.evidence["confidence"] < 0.6


def test_smoke_v1_judge_case_only_invokes_judge_for_the_unresolved_case():
    cases = load_dataset("smoke-v1")
    judge_cases = [case for case in cases if case.metadata.get("judge_rubric")]
    other_cases = [case for case in cases if case.id not in {item.id for item in judge_cases}]
    assert other_cases
    assert {case.id for case in judge_cases} == {
        "en-comp-judge-001",
        "en-comp-judge-nov-001",
        "en-comp-judge-low-001",
    }

    calls: list[str] = []

    def stub(item, model_output, rubric):
        calls.append(item["id"])
        return {"parsed_score": 1.0, "confidence": 0.95, "rationale": "inverse scope, one shared book"}

    cascade = Cascade([SchemaEvaluator(), RuleGraphEvaluator(), LlmJudgeEvaluator(judge_fn=stub)])

    for case in other_cases:
        gold = case.metadata["gold_structure"]["correct_choice"]
        outcome = cascade.evaluate(case, ApplicationOutput(status="success", raw_text=gold, output=gold))
        assert outcome.final_status == "pass", case.id

    for case in judge_cases:
        outcome = cascade.evaluate(
            case,
            ApplicationOutput(
                status="success",
                raw_text="The continuation forces inverse scope: one shared witness.",
                output="The continuation forces inverse scope: one shared witness.",
            ),
        )
        assert outcome.final_status == "pass", case.id
        assert outcome.deciding_evaluator == "llm_judge"

    assert calls == [case.id for case in judge_cases]
