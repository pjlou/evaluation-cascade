from pathlib import Path

import pytest

from evalcascade.store import ResultStore


@pytest.fixture
def tmp_store(tmp_path: Path) -> ResultStore:
    store = ResultStore(tmp_path / "evalcascade.sqlite")
    yield store
    store.close()
