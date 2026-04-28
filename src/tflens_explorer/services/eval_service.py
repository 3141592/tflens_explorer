"""Model loading service."""

import torch
from transformer_lens.model_bridge import TransformerBridge

def run_model_eval(model, eval):
    eval_name = eval['name']
    eval_prompt = eval['prompt']
    eval_next_tokens = eval['expected_next_tokens']

    context.session.current_prompt = eval_prompt
    breakpoint()
    return


def rank_expected_tokens():
    return

