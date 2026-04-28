"""Model loading service."""

import torch
from transformer_lens.model_bridge import TransformerBridge

def run_model_eval(model, eval):
    # get inference and next token
    results = model.generate(eval['prompt'], max_new_tokens=1, do_sample=False)

    # Get logit ranks and probablities
    eval_results = rank_expected_tokens(model, eval)
    eval_results["model_name"] = model.cfg.model_name

    return eval_results


def rank_expected_tokens(model, eval):
    eval_results = {}
    logits = model(eval['prompt'], prepend_bos=eval['prepend_bos'])

    # values=tensor([-82.2124, -82.5013, -82.8357, -83.0191, -83.1987, -83.3259, -83.4433, -83.5148, -83.5338, -83.6343], device='cuda:0',
    #               grad_fn=<TopkBackward0>),
    # indices=tensor([ 4314,  2323,  3996, 18507,   736,  5743,  1735,  2166, 34902,  7624],
    #               device='cuda:0'))
    final_logits = logits[0, -1]
    final_probs = torch.softmax(final_logits, dim=-1)
    topk_logits = torch.topk(final_logits, 10)
    topk_probs = torch.topk(final_probs, 10)

    str_token = model.to_str_tokens(topk_logits.indices[0])
    next_token_id = model.to_single_token(str_token)
    boolean_tensor = (topk_probs.indices == next_token_id) 
    next_token_rank = boolean_tensor.nonzero()

    #breakpoint()
    eval_results["next_token"] = str_token[0]
    eval_results["next_token_id"] = next_token_id
    eval_results["next_token_rank"] = next_token_rank.item()
    eval_results["next_token_score"] = round(topk_logits[0][0].item(), 2)
    eval_results["next_token_prob"] = round(topk_probs[0][0].item(), 2)
    eval_results['next_token_in_expected_tokens'] = eval_results['next_token'] in eval['expected_next_tokens']

    # Debugging
    for index in range(len(topk_probs.values)):
        str_token = model.to_str_tokens(topk_probs.indices[index])
        value = topk_probs.values[index]
        logit_line = f"[{index}] {value:.2f} -> '{str_token[0]}'"
        #print(logit_line)

    #breakpoint()
    return eval_results

