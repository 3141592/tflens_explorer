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
    eval_summary = {
        'expected_in_top_1': 0,
        'expected_in_top_5': 0,
        'total': 0,
    }
    for eval in evals:
        print(f"eval name: {eval['name']}")
        #print(run_model_eval(model, eval)))
        results = run_model_eval(model, eval)
        for k, v in results.items():
            print(f"{k}: {v}")

        print(f"Eval prompt: {eval['prompt']}")
        print(f"Expected next tokens:\n {eval['expected_next_tokens']}")
        print()

        eval_summary['model'] = results['model_name']
        eval_summary['total'] += 1
        eval_summary['expected_in_top_1'] += results['expected_in_top_1']
        eval_summary['expected_in_top_5'] += results['expected_in_top_5']

        breakpoint()
    return

    print("Eval Summary")
    print("------------")
    print("model: ")

