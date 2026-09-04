from evalcascade.models import CaseResult, RunMetadata, utcnow
from evalcascade.store import ResultStore


def test_store_roundtrip(tmp_store: ResultStore):
    meta = RunMetadata(
        run_id="abc123",
        created_at=utcnow(),
        dataset_version="smoke-v1",
        overall_status="pass",
    )
    case = CaseResult(
        run_id="abc123",
        case_id="en-agr-001a",
        raw_application_output="a",
        parsed_output="a",
        application_status="success",
        latency=0.1,
        final_status="pass",
        severity="medium",
        tags=["english"],
    )
    tmp_store.save_run(meta)
    tmp_store.save_case(case)
    loaded = tmp_store.load_run("abc123")
    assert loaded is not None
    assert loaded.dataset_version == "smoke-v1"
    cases = tmp_store.load_cases("abc123")
    assert cases[0].case_id == "en-agr-001a"
    assert tmp_store.list_runs()[0].run_id == "abc123"
