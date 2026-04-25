"""Prompt command handlers."""

from pathlib import Path
from tflens_explorer.core.types import CommandContext


def handle_tokens(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return
    
    from tflens_explorer.services.model_service import tokens
    token_list = tokens(model, prompt)

    for token in token_list:
        print(token)


def handle_logits(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return
    
    from tflens_explorer.services.model_service import logits
    logits_list = logits(model, prompt)

    for logit in logits_list:
        print(logit)


