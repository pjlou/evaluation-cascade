# src/schema/test_item.py
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ModuleType(str, Enum):
    ENGLISH = "english"
    FINNISH = "finnish"

class LanguageCode(str, Enum):
    EN = "en"
    FI = "fi"

class TierLevel(str, Enum):
    TIER_1_MORPHOLOGICAL = "tier_1_morphological"
    TIER_2_CLAUSAL = "tier_2_clausal"
    TIER_3_EXTENSION = "tier_3_extension"

class VerificationMethod(str, Enum):
    DEPENDENCY_PARSE = "dependency_parse"
    MORPHOLOGICAL_FEATURE = "morphological_feature"
    HYBRID_PARSE_MORPH = "hybrid_parse_morph"
    FORCED_CHOICE_TRUTH_CONDITIONAL = "forced_choice_truth_conditional"
    LEGACY_FORCED_TRUTH_CONDITIONAL = "forced_truth_conditional"

    @classmethod
    def _missing_(cls, value):
        if value == "forced_truth_conditional":
            return cls.FORCED_CHOICE_TRUTH_CONDITIONAL
        return None

class TestItem(BaseModel):
    id: str = Field(..., description="Unique ID, e.g., 'en-agr-001a' or 'fi-case-001a'")
    module: ModuleType
    tier: TierLevel
    phenomenon: str = Field(..., description="Target phenomenon, e.g., 'agreement_attraction', 'object_case_alternation'")
    language: LanguageCode
    prompt: str = Field(..., description="The prompt sent to the target LLM")
    
    # Gold structural expectation used by verifiers
    gold_structure: Dict[str, Any] = Field(
        ..., 
        description="Structural properties required for a pass (e.g., target head, expected case tag, condition ID)"
    )
    
    rule_node_id: str = Field(
        ..., 
        description="Grounding rule ID in NetworkX rule graph (e.g., 'RULE_FI_C1_NEGATION' or 'RULE_EN_AGR_HEAD')"
    )
    
    verification_method: VerificationMethod
    difficulty: str = Field(default="medium")
    source: str = Field(..., description="Citation source, e.g., 'Kiparsky 1998' or 'Bock & Miller 1991'")
    minimal_pair_of: Optional[str] = Field(None, description="ID of paired test item isolating this contrast")
    notes: Optional[str] = None

    class Config:
        use_enum_values = True