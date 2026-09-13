from pathlib import Path
import sys

VENDOR_ROOT = Path(__file__).resolve().parent.parent / "cognitive-eval" / "cognitive-eval-main"


def ensure_cognitive_eval_on_path() -> Path:
    """Make Cognitive-Eval's `src.*` imports resolvable without rewriting that tree."""
    root = str(VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return VENDOR_ROOT
