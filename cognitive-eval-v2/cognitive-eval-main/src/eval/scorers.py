# src/eval/scorers.py
from inspect_ai.scorer import Scorer, Score, Target, accuracy, stderr, scorer
from inspect_ai.solver import TaskState
from src.verifiers.english_verifiers import verify_english_agreement_attraction, verify_english_negation_scope
from src.verifiers.finnish_verifiers import verify_finnish_object_case, verify_finnish_negation_scope
from src.schema.rule_graph import build_v02_rule_graph
from src.cascade.dispatcher import evaluate
from src.cascade.ollama_judge import ollama_judge

# Initialize static rule graph for metadata enrichment
RULE_GRAPH = build_v02_rule_graph()


def _verify_negation_scope(model_output, gold_structure):
    """Keep the registry keyed by phenomenon while supporting both languages."""
    verifier = (
        verify_finnish_negation_scope
        if gold_structure.get("language") == "finnish"
        else verify_english_negation_scope
    )
    return verifier(model_output, gold_structure)


VERIFIER_REGISTRY = {
    "agreement_attraction": verify_english_agreement_attraction,
    "object_case_alternation": verify_finnish_object_case,
    "negation_scope": _verify_negation_scope,
}

@scorer(metrics=[accuracy(), stderr()])
def structural_linguistic_scorer() -> Scorer:
    """
    Inspect Scorer that executes deterministic verifiers based on item metadata.
    Enriches evaluation logs with formal rule graph explanations.
    """
    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata
        model_output = state.output.completion
        
        item = dict(metadata)
        item["gold_structure"] = dict(metadata.get("gold_structure", {}))
        item["gold_structure"]["language"] = metadata.get("module", metadata.get("language"))
        result = evaluate(
            item,
            model_output,
            verifier_registry=VERIFIER_REGISTRY,
            rule_graph=RULE_GRAPH,
            judge_fn=ollama_judge,
            judge_rubric=(
                "Assess whether the model output gives the correct answer for the item. "
                "Use the item metadata and return a score reflecting correctness."
            ),
        )

        return Score(
            value=result.score if result.score is not None else 0.0,
            answer=model_output,
            explanation=f"Cascade: {result.evaluator_name} | Result: {result.category}",
            metadata={
                "result": result.status.upper(),
                "cascade_stage": result.evaluator_name,
                "error_code": result.category,
                "cascade_evidence": result.evidence,
            }
        )
        
    return score