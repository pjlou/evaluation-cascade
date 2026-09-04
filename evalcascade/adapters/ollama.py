from __future__ import annotations

import time

from evalcascade.models import ApplicationOutput, EvaluationCase


class OllamaAdapter:
    name = "ollama"
    version = "1.0.0"

    def __init__(
        self,
        model: str,
        timeout_seconds: float = 30.0,
        retries: int = 1,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.temperature = temperature

    def run(self, case: EvaluationCase, config: dict) -> ApplicationOutput:
        try:
            import ollama
        except ImportError as exc:
            return ApplicationOutput(
                status="runtime_error",
                error=f"ollama package is not installed: {exc}",
            )

        attempts = max(self.retries, 0) + 1
        last_error: str | None = None
        prompt = case.input
        started = time.perf_counter()
        for attempt in range(attempts):
            try:
                response = ollama.chat(
                    model=config.get("model", self.model),
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": config.get("temperature", self.temperature)},
                )
                text = (response.get("message") or {}).get("content")
                if text is None or not str(text).strip():
                    last_error = "empty model response"
                    continue
                usage = None
                if isinstance(response, dict):
                    usage = {
                        key: response.get(key)
                        for key in ("eval_count", "prompt_eval_count", "eval_duration")
                        if key in response
                    }
                return ApplicationOutput(
                    status="success",
                    output=text,
                    raw_text=str(text),
                    latency_seconds=time.perf_counter() - started,
                    retries=attempt,
                    token_usage=usage or None,
                )
            except TimeoutError as exc:
                last_error = str(exc)
                if attempt + 1 >= attempts:
                    return ApplicationOutput(
                        status="timeout",
                        error=last_error,
                        latency_seconds=time.perf_counter() - started,
                        retries=attempt,
                    )
            except Exception as exc:  # noqa: BLE001 — adapters must record, not crash
                last_error = str(exc)
                name = type(exc).__name__.lower()
                if "timeout" in name or "timeout" in last_error.lower():
                    if attempt + 1 >= attempts:
                        return ApplicationOutput(
                            status="timeout",
                            error=last_error,
                            latency_seconds=time.perf_counter() - started,
                            retries=attempt,
                        )
                    continue
                if attempt + 1 >= attempts:
                    return ApplicationOutput(
                        status="runtime_error",
                        error=last_error,
                        latency_seconds=time.perf_counter() - started,
                        retries=attempt,
                    )
        return ApplicationOutput(
            status="retry_exhaustion",
            error=last_error or "retries exhausted",
            latency_seconds=time.perf_counter() - started,
            retries=attempts,
        )
