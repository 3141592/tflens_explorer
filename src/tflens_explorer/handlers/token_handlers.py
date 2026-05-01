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
    token_list = tokens(model, prompt, prepend_bos=True)
    context.session.prepend_bos = True

    for token in token_list:
        print(token)


def handle_token_decode(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    if len(context.args) == 0:
        print("Enter a token ID to decode. Use: token-decode 1234")
        return
    
    try:
        token_id = int(context.args[0])
    except:
        print(f"The token ID must be an integer. {context.args} is not valid.")
        return

    from tflens_explorer.services.model_service import token_decode
    decoded_token = token_decode(model, token_id)
    if decoded_token == -1:
        print("The token ID {token_id} cannot be larger than the model vocabulary {model.cfg.d_vocab}.")
    else:
        print(repr(decoded_token))

def handle_token_encode(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    if len(context.args) == 0:
        print("Enter a string to encode. Use: token-encode Emphasis")
        return

    str_token = context.args[0]

    from tflens_explorer.services.model_service import token_encode
    token_id = token_encode(model, str_token)
    print(token_id)
    

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
    logits_list = logits(model, prompt, prepend_bos=True)
    context.session.prepend_bos = True

    for logit in logits_list:
        print(logit)


