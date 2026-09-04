# src/schema/dataset_loader.py
import json
from pathlib import Path
from typing import List
from src.schema.test_item import TestItem

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

def load_dataset_by_phenomenon(module: str, phenomenon: str) -> List[TestItem]:
    """Loads and validates test items for a specific module and phenomenon file."""
    file_path = DATA_DIR / module / f"{phenomenon}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        raw_data = [raw_data]
        
    return [TestItem(**item) for item in raw_data]

def load_all_test_items() -> List[TestItem]:
    """Discovers and validates every test item in the data directory."""
    all_items = []
    for json_file in DATA_DIR.glob("**/*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            if isinstance(raw_data, dict):
                raw_data = [raw_data]
            all_items.extend([TestItem(**item) for item in raw_data])
    return all_items