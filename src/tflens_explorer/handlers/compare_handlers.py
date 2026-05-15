"""Eval command handlers."""

import statistics
from pathlib import Path
from tflens_explorer.core.types import CommandContext
from tflens_explorer.services.compare_service import compare

def handle_compare(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return

    return