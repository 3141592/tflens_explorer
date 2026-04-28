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
        eval_name = eval['name']
        eval_prompt = eval['prompt']
        eval_next_tokens = eval['expected_next_tokens']
        run_model_eval(model, eval_name, eval_prompt, eval_next_tokens)
    return

