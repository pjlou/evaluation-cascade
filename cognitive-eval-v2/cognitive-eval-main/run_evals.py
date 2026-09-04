# run_evals.py
import os
import subprocess
from inspect_ai import eval
from src.eval.tasks import convert_test_items_to_inspect_samples

def run_full_suite():
    print("=== Launching Cognitive-Eval Benchmark across 3 Model Families ===")
    
    # Set OLLAMA_MODELS to a comma-separated list to run only installed models.
    configured_models = os.getenv(
        "OLLAMA_MODELS",
        "qwen2.5:1.5b,qwen2.5:3b,qwen2.5:7b,ministral-3:8b,llama3.1:8b",
    )
    model_names = [model.strip() for model in configured_models.split(",") if model.strip()]
    installed_models = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, check=True
    ).stdout
    missing_models = [
        model for model in model_names
        if not any(line.split() and line.split()[0] == model for line in installed_models.splitlines()[1:])
    ]
    if missing_models:
        raise RuntimeError(
            "Ollama model(s) not installed: " + ", ".join(missing_models)
            + ". Install with `ollama pull <model>` or set OLLAMA_MODELS."
        )

    sample_count = len(convert_test_items_to_inspect_samples())
    limit = os.getenv("EVAL_LIMIT")
    token_limit = os.getenv("EVAL_TOKEN_LIMIT")
    ctl_server = os.getenv("EVAL_CTL_SERVER", "true").lower() in {"1", "true", "yes"}
    print(f"Models: {', '.join(model_names)}", flush=True)
    print(f"Samples per model: {limit or sample_count}", flush=True)
    print("Starting model requests; live progress follows:", flush=True)
    
    # Evaluate one model at a time, then ask Ollama to release its memory.
    for model_name in model_names:
        print(f"\nStarting {model_name}...", flush=True)
        try:
            eval(
                tasks="src/eval/tasks.py@cognitive_eval_benchmark",
                model=f"ollama/{model_name}",
                log_dir="./eval_logs",
                display="rich",
                log_realtime=True,
                ctl_server=ctl_server,
                max_subprocesses=1,
                limit=int(limit) if limit else None,
                token_limit=int(token_limit) if token_limit else None,
            )
        finally:
            subprocess.run(["ollama", "stop", model_name], check=False)
            print(f"Finished {model_name}; Ollama memory released.", flush=True)
    
    print("\n✓ Evaluation complete. Structured logs written to ./eval_logs")

if __name__ == "__main__":
    run_full_suite()