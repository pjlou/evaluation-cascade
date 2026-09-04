# src/cascade/schema.py
"""
Shared result schema for the verification cascade dispatcher.

Deliberately matches the EvaluationResult shape specified in the
Evaluation Cascade requirements doc (FR-3) field-for-field. The point of
building this here, inside Cognitive-Eval, is that it should be liftable
into Evaluation Cascade with no schema translation layer -- one of the two
concrete, reusable deliverables from this rescoped Phase B (the other is
the discovery/clustering module already built and tested).
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Status = Literal["pass", "fail", "review", "error", "not_applicable"]


@dataclass
class EvaluationResult:
    status: Status
    score: Optional[float]
    category: Optional[str]
    severity: Optional[str]
    evidence: dict[str, Any] = field(default_factory=dict)
    evaluator_name: str = ""
    evaluator_version: str = "1.0.0"
