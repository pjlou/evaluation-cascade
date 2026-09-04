import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

from utils import get_model_family, get_models_for_family, model_sort_key


def test_get_model_family_normalizes_qwen_variants():
    assert get_model_family("ollama/qwen2.5:7b") == "qwen2.5"
    assert get_model_family("qwen2.5:3b") == "qwen2.5"
    assert get_model_family("ollama/qwen-2.5:7b") == "qwen2.5"
    assert get_model_family("ollama/qwen2.5-7b") == "qwen2.5"
    assert get_model_family("ollama/qwen2.5:1.5b-instruct") == "qwen2.5"


def test_get_model_family_leaves_other_families_unchanged():
    assert get_model_family("ollama/ministral-3:8b") == "ministral-3"
    assert get_model_family("ollama/llama3.1:8b") == "llama3.1"


def test_get_models_for_family_returns_sorted_qwen_models():
    df = pd.DataFrame(
        {
            "model": [
                "ollama/qwen2.5:7b",
                "ollama/qwen2.5:1.5b",
                "ollama/qwen2.5:3b",
                "ollama/llama3.1:8b",
            ]
        }
    )

    assert get_models_for_family(df, "qwen2.5") == [
        "ollama/qwen2.5:1.5b",
        "ollama/qwen2.5:3b",
        "ollama/qwen2.5:7b",
    ]


def test_model_sort_key_orders_by_size_within_family():
    assert model_sort_key("ollama/qwen2.5:1.5b") < model_sort_key("ollama/qwen2.5:7b")
