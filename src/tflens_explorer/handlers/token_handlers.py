"""Prompt command handlers."""

import numbers
import torch
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

def handle_token_next(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return
    
    from tflens_explorer.services.model_service import logits_for_snapshot
    logits = logits_for_snapshot(model, prompt, prepend_bos=True)
    next_id = logits[0]['token_id']

    tokens = model.to_tokens(prompt, prepend_bos=True)
    new_tokens = torch.cat([
        tokens,
        torch.tensor([[next_id]], device=tokens.device)
    ], dim=1)

    next_token = model.to_string(new_tokens[-1][-1].item())
    context.session.current_prompt += next_token
    print(f"New prompt: {context.session.current_prompt}")

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


def handle_logits_for(context: CommandContext) -> None:
    if not context.args:
        print("Usage: logits-for \"<text>\" || <integer>")
        return

    str_token = context.args[0]

    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return

    from tflens_explorer.services.model_service import logits_for
    logits_list = logits_for(model, prompt, str_token, prepend_bos=True)
    context.session.prepend_bos = True

    for logit in logits_list:
        print(logit)

    print()


def handle_embedding_similarity(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    if len(context.args) != 2:
        print("Specify two tokens to compare. Use: embedding-similarity <word1> <word2>")
        return

    word1 = context.args[0]
    word2 = context.args[1]
     
    from tflens_explorer.services.model_service import embedding_cosine_similarity

    cos_sim = embedding_cosine_similarity(model, word1, word2)

    if isinstance(cos_sim, str):
        print(f"Cosine similarity for '{word1}' and '{word2}' could not be calculated.")
    elif isinstance(cos_sim.item(), numbers.Number):
        print(f"Cosine similarity for '{word1}' and '{word2}': {round(cos_sim.item(), 4)}")
    else:
        print(f"Cosine similarity for '{word1}' and '{word2}' could not be calculated.")
