"""Model loading service."""

import torch
from transformer_lens.model_bridge import TransformerBridge

def run_model_eval(model, eval_name, eval_prompt, eval_next_tokens):

    results = model.generate(eval_prompt, max_new_tokens=1, do_sample=False)
    breakpoint()
    return


def rank_expected_tokens():
    return

