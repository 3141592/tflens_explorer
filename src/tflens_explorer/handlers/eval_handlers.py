"""Eval command handlers."""

from pathlib import Path
from tflens_explorer.core.types import CommandContext

def handle_eval_run(context) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    from tflens_explorer.config.config_loader import load_model_evals
    from tflens_explorer.services.eval_service import run_model_eval

    evals = load_model_evals()
    
    for eval in evals:
        print(f"eval name: {eval['name']}")
        print(run_model_eval(model, eval))
        print(f"Eval prompt: {eval['prompt']}")
        print(f"Expected next tokens:\n {eval['expected_next_tokens']}")
        print()
    return

