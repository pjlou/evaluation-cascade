"""Ollama-backed Cascade Stage 3 judge."""

import json
import os
from typing import Any


def ollama_judge(item: dict[str, Any], model_output: str, rubric: str) -> dict[str, Any]:
    """Judge an otherwise unresolved item and return structured evidence."""
    import ollama

    judge_model = os.getenv("OLLAMA_JUDGE_MODEL", "llama3.1:8b")
    prompt = (
        "Evaluate the model output against the rubric. Return JSON only with "
        'keys "parsed_score" (number from 0 to 1), "confidence" (number from 0 to 1), '
        'and "rationale" (string).\n\n'
        f"Rubric:\n{rubric}\n\n"
        f"Item metadata:\n{json.dumps(item, ensure_ascii=True, sort_keys=True)}\n\n"
        f"Model output:\n{model_output}"
    )
    response = ollama.chat(
        model=judge_model,
        messages=[
            {
                "role": "system",
                "content": "You are a careful evaluation judge. Follow the requested JSON schema exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        format="json",
    )
    raw_output = response["message"]["content"] if isinstance(response, dict) else response.message.content
    parsed = json.loads(raw_output)
    parsed_score = float(parsed["parsed_score"])
    confidence = float(parsed["confidence"])
    if not 0 <= parsed_score <= 1 or not 0 <= confidence <= 1:
        raise ValueError("Judge score and confidence must be between 0 and 1")

    return {
        "judge_model": judge_model,
        "judge_prompt": prompt,
        "rubric": rubric,
        "raw_output": raw_output,
        "parsed_score": parsed_score,
        "confidence": confidence,
        "rationale": str(parsed.get("rationale", "")),
    }