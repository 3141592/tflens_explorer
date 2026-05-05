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
    topk_logits = torch.topk(final_logits, 5)
    topk_probs = torch.topk(final_probs, 5)

    str_token = model.to_str_tokens(topk_logits.indices[0])
    next_token_id = model.to_single_token(str_token)
    boolean_tensor = (topk_probs.indices == next_token_id) 
    next_token_rank = boolean_tensor.nonzero()

    # Get ranks of the expected token(s)
    sorted_probs, sorted_indices = torch.sort(final_probs, descending=True)
    expected_token_ranks = []

    for token in eval["expected_next_tokens"]:
        print("TOKEN CHECK: ", token)
        token_id = model.to_single_token(token)
        matches = (sorted_indices == token_id).nonzero(as_tuple=True)[0]
        rank = matches.item() + 1 if len(matches) else None
        expected_token_ranks.append({
            "token": token,
            "token_id": token_id,
            "rank": rank,
            "prob": final_probs[token_id].item(),
        })

    found_ranks = [
        item["rank"]
        for item in expected_token_ranks
        if item["rank"] is not None
    ]
    expected_token_best_rank = min(found_ranks) if found_ranks else None

    expected_in_top_1 = expected_token_best_rank == 1
    expected_in_top_5 = (
        expected_token_best_rank is not None
        and expected_token_best_rank <= 5
    )
    eval_results["next_token"] = str_token[0]
    eval_results["next_token_id"] = next_token_id
    eval_results["next_token_score"] = round(topk_logits[0][0].item(), 2)
    eval_results["next_token_prob"] = round(topk_probs[0][0].item(), 2)
    eval_results['expected_in_top_1'] = expected_in_top_1
    eval_results['expected_in_top_5'] = expected_in_top_5
    eval_results['expected_token_best_rank'] = expected_token_best_rank
 
    # Debugging
    for index in range(len(topk_probs.values)):
        str_token = model.to_str_tokens(topk_probs.indices[index])
        value = topk_probs.values[index]
        logit_line = f"[{index}] {value:.2f} -> '{str_token[0]}'"
        #print(logit_line)

    #breakpoint()
    return eval_results

