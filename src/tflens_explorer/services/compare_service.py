"""Compare Service"""

import os
import yaml
import torch
import re
import traceback
import datetime
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tflens_explorer.services.model_service import tokens_for_snapshot, logits_for_snapshot, tokens_shape, logits_shape
from tflens_explorer.services.model_service import cache_summary_for_snapshot, get_model_alias
from tflens_explorer.services.model_service import cache_summary_for_snapshot_all
from tflens_explorer.core.types import CommandContext
from tflens_explorer.cli.utilities import get_shape
from tflens_explorer.core.snapshot_types import Snapshot, SNAPSHOT_PATH, SNAPSHOT_DATA_PATH
from tflens_explorer.core.snapshot_types import SnapshotMetadata, verify_snapshot
from tflens_explorer.core.snapshot_types import CacheSummary, Model

def snapshot_create(context: CommandContext, snapshot_name: str, hook: str) -> None:
    """Create a model comparison snapshot."""
    model_name = context.session.current_model_name
    current_model = context.session.model
    prompt = context.session.current_prompt
    prepend_bos = context.session.prepend_bos
    model_alias = get_model_alias(model_name)

    if len(model_alias) > 0:
        model_name = f"{model_alias} -> {model_name}"

    metadata = SnapshotMetadata(
        name=snapshot_name,
        creation_date=datetime.datetime.now().strftime("%B %d, %Y %I:%M%p")
    )

    model = Model(
        name=model_name,
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        num_ctx=current_model.cfg.n_ctx,
        prepend_bos=context.session.prepend_bos,
        layers=current_model.cfg.n_layers,
        heads=current_model.cfg.n_heads,
        vocabulary=current_model.cfg.d_vocab,
    )
    
    try:
        if 'all' == hook:
            cache = cache_summary_for_snapshot_all(current_model, prompt, snapshot_name)
        else:
            cache = cache_summary_for_snapshot(current_model, prompt, hook, snapshot_name)
    except Exception as ex:
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        return

    snapshot = Snapshot(
        metadata=metadata,
        model=model,
        prompt=prompt,
        token_shape=tokens_shape(current_model, prompt, prepend_bos),
        tokens=tokens_for_snapshot(current_model, prompt, prepend_bos),
        logit_shape=logits_shape(current_model, prompt, prepend_bos),
        logits=logits_for_snapshot(current_model, prompt, prepend_bos),
        cache=cache,
    )
    
    snapshot.save()

def snapshots_list():
    directory = Path(SNAPSHOT_PATH)
    directories = [item for item in directory.iterdir() if item.is_dir()]

    for item in directories:
        print(str(item).split("/")[-1])

def compare_runs():
    print("compare-runs")

def compare_mlp():
    print("compare-mlp")

def compare_attention():
    print("compare-attention")

def compare_generated():
    print("compare-generated")

def compare_evals():
    print("compare-evals")

def compare_logits(snapshot1: Snapshot, snapshot2: Snapshot):
    all_args = locals()
    for name, value in all_args.items():
        if verify_snapshot(value):
            pass
        else:
            print(f"Snapshot {value} does not exist. Use: snapshots-list to find valid snapshots.")
            return

    snapshot1 = Snapshot(name=snapshot1)
    snapshot1.load()

    snapshot2 = Snapshot(name=snapshot2)
    snapshot2.load()

    logits_1_size = snapshot1.logits[0]
    logits_2_size = snapshot2.logits[0]
    logit_size_comparison = logits_1_size == logits_2_size
    logit_comparison = compare_logits_details(snapshot1.logits, snapshot2.logits)

    print(f"Models:")
    print(f"  A: {snapshot1.model.name}")
    print(f"  B: {snapshot2.model.name}")
    print()
    print(f"Prompt:")
    print(f"  A: {snapshot1.prompt}")
    print(f"  B: {snapshot2.prompt}")
    print()
    print(f"Logits:")
    print(f"  same length: {logit_size_comparison}")
    print(f"    A: {str(logits_1_size)}")
    print(f"    B: {str(logits_2_size)}")
    print(f"  top-1:")
    print(f"    A: {logit_comparison[0][0]}")
    print(f"    B: {logit_comparison[0][1]}")
    print(f"  top-5 overlap: {logit_comparison[1]}/5")
    print(f"  rankings:")
    compare_logits_ranks(snapshot1.logits, snapshot2.logits)
    print(f"  probabilities:")
    compare_logits_probs(snapshot1.logits, snapshot2.logits)



def compare_logits_details(logits1, logits2):
    logits1.pop(0)
    logits2.pop(0)
    top_1 = [logits1[0]['token']]
    top_1.append(logits2[0]['token'])
    top_5_match_count = 0
    for index, item in enumerate(logits1):
        for index2, item2 in enumerate(logits2):
            if item['token'] == item2['token']:
                top_5_match_count += 1
            if index == 4:
                pass
        if index == 4:
            break
    return top_1, top_5_match_count

def compare_cache(snapshot1: Snapshot, snapshot2: Snapshot):
    all_args = locals()
    for name, value in all_args.items():
        if verify_snapshot(value):
            pass
        else:
            print(f"Snapshot {value} does not exist. Use: snapshots-list to find valid snapshots.")
            return

    snapshot1 = Snapshot(name=snapshot1)
    snapshot1.load()

    snapshot2 = Snapshot(name=snapshot2)
    snapshot2.load()

    print(f"Models:")
    print(f"  A: {snapshot1.model.name}")
    print(f"  B: {snapshot2.model.name}")
    print()
    print(f"Prompt:")
    print(f"  A: {snapshot1.prompt}")
    print(f"  B: {snapshot2.prompt}")
    print()
    print(f"Cache summary:")
    cache_activation_summary(snapshot1.cache, snapshot2.cache)
    #cache_activation_comparison(snapshot1.cache, snapshot2.cache)

def cache_activation_comparison(cache1, cache2):
    print(f"    {'Attribute':<10} {'A':>30} {'B':>30}")
    for activation1, activation2 in zip(cache1, cache2):
        length = min(len(list(activation1.keys())), len(list(activation1.keys())))
        for index in range (length - 1):
            key1 = list(activation1.keys())[index]
            value1 = list(activation1.values())[index]
            key2 = list(activation2.keys())[index]
            value2 = list(activation2.values())[index]

            if key1 == "value" or key2 == "value":
                continue

            if key1 == key2:
                print(f"    {key1:<10} {value1:>30} {value2:>30}")
            else:
                print(f"Cache activation keys {key1} and {key2} do not match. Check the snapshot cache data.")
                print()
                return
        print()

    print()
    return

def compare_snapshots(snapshot1_name: Snapshot, snapshot2_name: Snapshot, diff, percent):
    all_args = locals()
    for name, value in all_args.items():
        if name in ["diff","percent"]:
            continue
        if verify_snapshot(value):
            pass
        else:
            print(f"Snapshot {value} does not exist. Use: snapshots-list to find valid snapshots.")
            return

    snapshot1 = Snapshot.load(snapshot1_name)
    snapshot2 = Snapshot.load(snapshot2_name)

    token_size_comparison = (snapshot1.token_shape == snapshot2.token_shape)
    token_id_comparison = compare_token_ids(snapshot1.tokens, snapshot2.tokens)
    token_comparison = compare_tokens(snapshot1.tokens, snapshot2.tokens)
    logit_size_comparison = (snapshot1.logit_shape == snapshot2.logit_shape)
    logit_comparison = compare_logits_details(snapshot1.logits, snapshot2.logits)

    print(f"Models:")
    print(f"  A: {snapshot1.model.name}")
    print(f"  B: {snapshot2.model.name}")
    print()
    print(f"Prompt:")
    print(f"  A: {snapshot1.prompt}")
    print(f"  B: {snapshot2.prompt}")
    print()
    print(f"Tokenization:")
    print(f"  same length: {token_size_comparison}")
    print(f"  differing token_ids:")
    print_token_id_comparison(snapshot1, snapshot2, token_id_comparison)
    print()
    print(f"  differing tokens:")
    print_token_comparison(snapshot1, snapshot2, token_comparison)
    print()
    print(f"Logits:")
    print(f"  same length: {logit_size_comparison}")
    print(f"    A: {str(snapshot1.logit_shape)}")
    print(f"    B: {str(snapshot2.logit_shape)}")
    print(f"  top-1:")
    print(f"    A: {logit_comparison[0][0]}")
    print(f"    B: {logit_comparison[0][1]}")
    print(f"  top-5 overlap: {logit_comparison[1]}/5")
    print()
    if len(snapshot1.cache) > 0 and len(snapshot2.cache) > 0:
        print(f"Cache activation differences (usable finite values):")
        cache_activation_summary(snapshot1, snapshot2, diff, percent)
        #if snapshots_have_raw_cache_values(snapshot1.cache, snapshot2.cache):
        #    cache_activation_summary_2(snapshot1.cache, snapshot2.cache)
        #else:
        #    print("Raw cache tensor values not available; skipping tensor-diff summary.")

def compare_token_ids(tokens1, tokens2):
    tokens1.pop(0)
    tokens2.pop(0)
    token_id_comparison = [None] * max(len(tokens1), len(tokens2))
    for index, item in enumerate(tokens1):
        if len(tokens2) <= index:
            token_id_comparison[index] = False
        elif item.token_id == tokens2[index].token_id:
            token_id_comparison[index] = True
        else:
            token_id_comparison[index] = False
    for index, item in enumerate(tokens2):
        try:
            if len(tokens1) <= index:
                token_id_comparison[index] = False
            elif item.token_id == tokens1[index].token_id:
                token_id_comparison[index] = True
            else:
                token_id_comparison[index] = False
        except:
            traceback.print_exception(type(ex), ex, ex.__traceback__)

    return token_id_comparison

def print_token_id_comparison(snapshot1, snapshot2, token_id_comparison):
    for index, item in enumerate(token_id_comparison):
        if item:
            pass
        elif (len(snapshot1.tokens) <= index) \
             and (len(snapshot1.tokens) <= index):
             pass
        else:
            if len(snapshot1.tokens) > index:
                print(f"    A: {snapshot1.tokens[index]}")
            else:
                print(f"    A: NA")

            if len(snapshot2.tokens) > index:
                print(f"    B: {snapshot2.tokens[index]}")
            else:
                print(f"    B: NA")

    if all(token_id_comparison):
        print(f"    All token_ids are the same.")

    return

def compare_tokens(tokens1, tokens2):
    token_comparison = [None] * max(len(tokens1), len(tokens2))
    for index, item in enumerate(tokens1):
        if len(tokens2) <= index:
            token_comparison[index] = False
        elif item.token == tokens2[index].token:
            token_comparison[index] = True
        else:
            token_comparison[index] = False

    for index, item in enumerate(tokens2):
        try:
            if len(tokens1) <= index:
                token_comparison[index] = False
            elif item.token == tokens1[index].token:
                token_comparison[index] = True
            else:
                token_comparison[index] = False
        except:
            traceback.print_exception(type(ex), ex, ex.__traceback__)

    return token_comparison

def print_token_comparison(snapshot1, snapshot2, token_comparison):
    for index, item in enumerate(token_comparison):
        if item:
            pass
        elif (len(snapshot1.tokens) <= index) \
             and (len(snapshot1.tokens) <= index):
             pass
        else:
            if len(snapshot1.tokens) > index:
                print(f"    A: {snapshot1.tokens[index]}")
            else:
                print(f"    A: NA")

            if len(snapshot2.tokens) > index:
                print(f"    B: {snapshot2.tokens[index]}")
            else:
                print(f"    B: NA")

    if all(token_comparison):
        print(f"    All tokens are the same.")

    return

def compare_models():
    print("compare-models")

def compare_logits_ranks(logits1, logits2):
    rankings = []
    for item1 in enumerate(logits1):
        item1 = item1[1]
        for item2 in enumerate(logits2):
            item2 = item2[1]
            if item2['token'] == item1['token']:
                ranking = {
                    'index1': item1['index'],
                    'prob1': item1['prob'],
                    'token1': item1['token'],
                    'index2': item2['index'],
                    'prob2': item2['prob'],
                    'token2': item2['token'],
                }
                rankings.append(ranking)

    print(f"    {'Token':<20} {'Rank A':>7} {'Rank B':>7} {'ΔRank':>7}")
    if len(rankings) > 0:
        for ranking in rankings:
            token = ranking['token1']
            token = token.replace("\n", "\\n")[:20]
            index1 = ranking['index1']
            index2 = ranking['index2']
            delta = index1 - index2
            print(f"    {token:<20} {index1:>7} {index2:>7} {delta:>7}")
    else:
        print(f"    No tokens in common.")
    
    print()
    return

def compare_logits_probs(logits1, logits2):
    rankings = []
    for item1 in enumerate(logits1):
        item1 = item1[1]
        for item2 in enumerate(logits2):
            item2 = item2[1]
            if item2['token'] == item1['token']:
                ranking = {
                    'index1': item1['index'],
                    'prob1': item1['prob'],
                    'token1': item1['token'],
                    'index2': item2['index'],
                    'prob2': item2['prob'],
                    'token2': item2['token'],
                }
                rankings.append(ranking)

    print(f"    {'Token':<20} {'Prob A':>7} {'Prob B':>7} {'ΔProb':>7}")
    if len(rankings) > 0:
        for ranking in rankings:
            token = ranking['token1']
            token = token.replace("\n", "\\n")[:20]
            prob1 = ranking['prob1']
            prob2 = ranking['prob2']
            delta = round(prob1 - prob2, 2)
            print(f"    {token:<20} {prob1:>7} {prob2:>7} {delta:>7}")
    else:
        print(f"    No tokens in common.")
    
    print()
    return

def cache_activation_summary(snapshot1, snapshot2, diff, percent):
    if diff == None:
        diff = 0
    if percent == None:
        percent = 0

    # Create new file for saving cosine similarities for each layer and head
    filename = f"{snapshot1.metadata.name}_vs_{snapshot2.metadata.name}"
    open(SNAPSHOT_DATA_PATH / filename, 'w')

    print(
        f"    {'A/B':<4}"
        f"{'hook_name':<36}"
        f"{'shape':>15}"
        f"{'min':>13}"
        f"{'max':>13}"
        f"{'mean':>13}"
        f"{'mean_abs_diff':>16}"
        f"{'cos_sim':>16}"
    )

    different_values_count = 0
    hook_count = 0
    seen = set()
    for activation1, activation2 in zip(snapshot1.cache, snapshot2.cache):
        hook1 = activation1.hook
        hook2 = activation2.hook
        shape1 = get_shape(activation1.shape)
        shape2 = get_shape(activation2.shape)
        minimum1 = activation1.minimum
        minimum2 = activation2.minimum
        maximum1 = activation1.maximum
        maximum2 = activation2.maximum

        mean_abs_diff = cache_mean_abs_diff(snapshot1.cache_tensors[activation1.hook],
                                            snapshot2.cache_tensors[activation2.hook])

        cosine_similarity = cache_cosine_similarity(snapshot1.cache_tensors[activation1.hook],
                                                    snapshot2.cache_tensors[activation1.hook])

        if cosine_similarity is None:
            cos_similarity_str = "n/a"
        else:
            cos_similarity_str = f"{cosine_similarity:>14.6f}"

        include_diff = cache_diff(mean_abs_diff, diff)
        include_percent = cache_percent_diff(activation1.mean, activation2.mean, percent)
        if include_diff and include_percent:
            pass
        else:
            continue

        mean1_str = (
            activation1.mean
            if isinstance(activation1.mean, str)
            else f"{activation1.mean:.4f}"
        )

        mean2_str = (
            activation2.mean
            if isinstance(activation2.mean, str)
            else f"{activation2.mean:.4f}"
        )

        if hook1 != hook2:
            #print(f"Cache hooks {hook1} and {hook2} do not match.")
            continue

        if (minimum1 == minimum2) and (maximum1 == maximum2):
            continue
        else:
            different_values_count += 1

        # de-duplicate results
        mean_abs_diff_key = (
            None
            if mean_abs_diff is None
            else round(mean_abs_diff, 6)
        )

        key = (
            activation1.hook,
            mean1_str,
            mean2_str,
            mean_abs_diff_key,
        )

        mean_abs_diff_str = (
            "n/a"
            if mean_abs_diff is None
            else f"{mean_abs_diff:>14.4f}"
        )

        if key in seen:
            continue
        seen.add(key)
        
        hook_count += 1

        print(
            f"    {'A:':<4}"
            f"{hook1:<35} "
            f"{shape1:>15} "
            f"{minimum1:>12.4f} "
            f"{maximum1:>12.4f} "
            f"{mean1_str:>12}"
            f"{mean_abs_diff_str:>16}"
            f"{cos_similarity_str:>16}"
        )
                
        print(
            f"    {'B:':<4}"
            f"{hook2:<35} "
            f"{shape2:>15} "
            f"{minimum2:>12.4f} "
            f"{maximum2:>12.4f} "
            f"{mean2_str:>12}"
            f"{'':>16}"
        )

        cache_cosine_similarity_per_head(snapshot1,
                                         snapshot2,
                                         activation1,
                                         activation2)
        print()

    print()
    print(f"  Total hooks listed: {hook_count}")
    
    if different_values_count == 0:
        print("    No cache summary differences found.")

    print()

    # Create graph of cosine similarity changes
    plot_cosine_chart(filename)
    plot_cosine_chart2(filename)
    plot_cosine_chart3(filename)
    plot_cosine_chart4(filename)

    return

# 
# Check to see if activatiioni percent difference is greater than supplied limit
# True: show the hook row
# False: exclude the hook row
def cache_percent_diff(mean1, mean2, percent_limit):
    percent_limit = abs(percent_limit)

    if percent_limit == 0:
        return True

    diff = abs(mean1 - mean2)

    if diff == 0:
        return False

    if mean1 == 0 or mean2 == 0:
        percent_diff = diff * 100
    else:
        percent_diff = (diff * 100) / mean2
        
    #print(f"{percent_limit}, {abs(mean1-mean2)}, {abs(mean1-mean2) / mean2}")
    return (percent_diff > percent_limit)

# 
# Check to see if activatiion difference is greater than supplied limit
# True: show the hook row
# False: exclude the hook row
def cache_diff(mean_abs_diff, diff_limit):
    if diff_limit == 0:
        return True
    return (mean_abs_diff > diff_limit)

def cache_mean_abs_diff(tensor1, tensor2):
    v1 = tensor1.detach().flatten()
    v2 = tensor2.detach().flatten()

    if v1.numel() != v2.numel():
        return None

    if not v1.is_floating_point():
        v1 = v1.float()
    if not v2.is_floating_point():
        v2 = v2.float()

    mask = finite_reasonable_mask(v1, v2)

    if mask.sum().item() == 0:
        return None

    return (v1[mask] - v2[mask]).abs().mean().item()

def cache_cosine_similarity(tensor1, tensor2):
    v1 = tensor1.detach().flatten()
    v2 = tensor2.detach().flatten()

    if v1.numel() != v2.numel():
        return None

    if not v1.is_floating_point():
        v1 = v1.float()
    if not v2.is_floating_point():
        v2 = v2.float()

    if not is_cosine_eligible(v1, v2):
        return None

    mask = cache_value_mask(v1, v2)

    if mask.sum().item() == 0:
        return None

    v1_masked = v1[mask]
    v2_masked = v2[mask]

    if v1_masked.norm().item() == 0 or v2_masked.norm().item() == 0:
        return None

    cosine_sim_raw = torch.nn.functional.cosine_similarity(
        v1_masked,
        v2_masked,
        dim=0,
    )

    return cosine_sim_raw.item()

def cache_cosine_similarity_per_head(snapshot1, snapshot2, activation1, activation2):
    tensor1 = snapshot1.cache_tensors[activation1.hook]
    tensor2 = snapshot2.cache_tensors[activation2.hook]

    if not tensor1.is_floating_point():
        tensor1 = tensor1.float()

    if not tensor2.is_floating_point():
        tensor2 = tensor2.float()

    heads1 = snapshot1.model.heads
    heads2 = snapshot2.model.heads
    heads = min(heads1, heads2)

    if is_cosine_eligible(tensor1, tensor2):
        head_axis = head_axis_for_hook(activation1.hook, tensor1, heads1)

        if head_axis is not None:
            for head in range(heads):
                if head_axis == 0:
                    v1 = tensor1[head, :, :].reshape(-1)
                    v2 = tensor2[head, :, :].reshape(-1)
                elif head_axis == 1:
                    v1 = tensor1[:, head, :].reshape(-1)
                    v2 = tensor2[:, head, :].reshape(-1)
                else:
                    raise ValueError(f"Unsupported head axis: {head_axis}")

                sim = cache_cosine_similarity(v1, v2)
                save_cosine_similarity_data(snapshot1.metadata.name, snapshot2.metadata.name, 
                                            activation1.hook, activation2.hook, head, sim)
                 
                if sim < 0.90:
                    print(f"            head{head} cos_sim: {round(sim, 4)}")

def save_cosine_similarity_data(name1, name2, hook1, hook2, head, sim):
    filename = f"{name1}_vs_{name2}"
    with open(SNAPSHOT_DATA_PATH / filename, 'a') as f:
        f.write(f"{hook1},{hook2}, {head},{sim}\n")

def cache_value_mask(v1, v2, max_abs=1e20):
    return (
        torch.isfinite(v1)
        & torch.isfinite(v2)
        & (v1.abs() < max_abs)
        & (v2.abs() < max_abs)
    )

def finite_reasonable_mask(v1, v2, max_abs=1e20):
    return (
        torch.isfinite(v1)
        & torch.isfinite(v2)
        & (v1.abs() < max_abs)
        & (v2.abs() < max_abs)
    )

def is_cosine_eligible(tensor1, tensor2) -> bool:
    return (
        tensor1.shape == tensor2.shape
        and tensor1.numel() > 0
        and tensor1.is_floating_point()
        and tensor2.is_floating_point()
    ) 

def head_axis_for_hook(hook_name: str, tensor, n_heads: int) -> int | None:
    if tensor.ndim == 3 and tensor.shape[1] == n_heads:
        # [seq, head, d_head]
        return 1

    if tensor.ndim == 3 and tensor.shape[0] == n_heads:
        # [head, query, key]
        return 0

    return None

def snapshots_have_raw_cache_values(cache1, cache2):
    if not cache1 or not cache2:
        return False

def plot_cosine_chart(filename: str) -> None:
    """Read cosine similarity per-head CSV data and plot a line-segment chart.

    The CSV is expected at ``snapshots/data/<filename>`` with the format::

        hook1, hook2, head, cosine_similarity

    For each attention head a piecewise-linear path is drawn.  Each segment
    spans one unit on the x-axis and its *slope* is set to
    ``arccos(cosine_similarity)``.  Cosine similarity close to 1 therefore
    produces a flat segment, while a low similarity produces a steep upward
    segment.

    The output image is saved to ``snapshots/data/<filename>.png``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    filepath = SNAPSHOT_DATA_PATH / filename
    if not filepath.is_file():
        print(f"File not found: {filepath}")
        return

    # ── read CSV ──────────────────────────────────────────────────────
    rows: list[tuple[str, int, float]] = []     # (hook, head, cos_sim)
    seen_hooks: list[str] = []
    seen_heads: set[int] = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            hook = parts[0].strip()
            head = int(parts[2].strip())
            cos_sim = float(parts[3].strip())
            rows.append((hook, head, cos_sim))
            if hook not in seen_hooks:
                seen_hooks.append(hook)
            seen_heads.add(head)

    if not rows:
        print("No data rows found.")
        return

    # Build (hook_index, cos_sim) per head, preserving file order
    hook_index_of = {h: i for i, h in enumerate(seen_hooks)}
    heads: dict[int, list[tuple[int, float]]] = {h: [] for h in seen_heads}
    for hook, head, cos_sim in rows:
        heads[head].append((hook_index_of[hook], cos_sim))

    # ── draw ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    n_heads = len(heads)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_heads))
    
    for (head, segments), color in zip(
        sorted(heads.items()), colors
    ):
        x = 0.0
        y = 0.0
        xs: list[float] = [x]
        ys: list[float] = [y]
        for _hook_idx, cos_sim in sorted(segments):
            angle = math.acos(max(-1.0, min(1.0, cos_sim)))  # clamp to [-1, 1]
            x += 1.0
            y += angle
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, color=color, label=f"Head {head}")

    ax.set_xlabel("Hook index (each unit = one hook)")
    ax.set_ylabel("Cumulative arccos(cosine similarity)")
    ax.set_title(f"Cosine similarity change per head — {filename}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

    # Tick labels – abbreviated hook names
    index = 0
    for item in seen_hooks:
        seen_hooks[index] = item.replace("blocks.", "D")
        index += 1

    short_names = [
        #h.split(".attn.")[0] + "." + h.split(".attn.")[1][:8] if ".attn." in h else h
        h.split(".attn.")[0] + "." + h.split(".attn.")[1] if ".attn." in h else h
        for h in seen_hooks
    ]

    ax.set_xticks(range(len(seen_hooks)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)

    fig.tight_layout()

    outpath = SNAPSHOT_DATA_PATH / f"{filename}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {outpath}")


def plot_cosine_chart2(filename: str) -> None:
    """Read cosine similarity per-head CSV data and plot a heatmap.

    The CSV is expected at ``snapshots/data/<filename>`` with the format::

        hook1, hook2, head, cosine_similarity

    The heatmap uses hooks as the x-axis and heads as the y-axis, with
    cosine similarity shown as cell color.

    The output image is saved to ``snapshots/data/<filename>.png``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    filepath = SNAPSHOT_DATA_PATH / filename
    if not filepath.is_file():
        print(f"File not found: {filepath}")
        return

    # ── read CSV ──────────────────────────────────────────────────────
    rows: list[tuple[str, int, float]] = []     # (hook, head, cos_sim)
    seen_hooks: list[str] = []
    seen_heads: set[int] = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            hook = parts[0].strip()
            head = int(parts[2].strip())
            cos_sim = float(parts[3].strip())
            rows.append((hook, head, cos_sim))
            if hook not in seen_hooks:
                seen_hooks.append(hook)
            seen_heads.add(head)

    if not rows:
        print("No data rows found.")
        return

    # ── build heatmap matrix ──────────────────────────────────────────
    hook_index_of = {h: i for i, h in enumerate(seen_hooks)}
    n_hooks = len(seen_hooks)
    sorted_heads = sorted(seen_heads)
    n_heads = len(sorted_heads)
    head_index_of = {h: i for i, h in enumerate(sorted_heads)}

    # NaN fill so missing combos show as a distinct color (via cmap.set_bad)
    matrix = np.full((n_heads, n_hooks), np.nan)
    for hook, head, cos_sim in rows:
        matrix[head_index_of[head], hook_index_of[hook]] = cos_sim

    # ── draw heatmap ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_hooks * 0.6), max(4, n_heads * 0.5)))

    cmap = plt.cm.RdYlGn  # red (low) → yellow → green (high)
    cmap.set_bad(color='lightgray')

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", rotation=270, labelpad=15)

    # Tick labels – abbreviated hook names
    index = 0
    for item in seen_hooks:
        seen_hooks[index] = item.replace("blocks.", "D")
        index += 1

    short_names = [
        #h.split(".attn.")[0] + "." + h.split(".attn.")[1][:8] if ".attn." in h else h
        h.split(".attn.")[0] + "." + h.split(".attn.")[1] if ".attn." in h else h
        for h in seen_hooks
    ]
    ax.set_xticks(range(n_hooks))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"Head {h}" for h in sorted_heads], fontsize=8)

    ax.set_xlabel("Hook")
    ax.set_ylabel("Head")
    ax.set_title(f"Cosine similarity heatmap — {filename}")

    # Annotate cells with values if grid is not too large
    if n_hooks * n_heads <= 200:
        for i in range(n_heads):
            for j in range(n_hooks):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="black")

    fig.tight_layout()

    outpath = SNAPSHOT_DATA_PATH / f"{filename}2.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to {outpath}")


def plot_cosine_chart3(filename: str) -> None:
    """Read cosine similarity per-head CSV data and plot a heatmap.

    The CSV is expected at ``snapshots/data/<filename>`` with the format::

        hook1, hook2, head, cosine_similarity

    The heatmap uses hooks as the x-axis and heads as the y-axis, with
    cosine similarity shown as cell color.

    The output image is saved to ``snapshots/data/<filename>.png``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    filepath = SNAPSHOT_DATA_PATH / filename
    if not filepath.is_file():
        print(f"File not found: {filepath}")
        return

    # ── read CSV ──────────────────────────────────────────────────────
    rows: list[tuple[str, int, float]] = []     # (hook, head, cos_sim)
    seen_hooks: list[str] = []
    seen_heads: set[int] = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            hook = parts[0].strip()
            if "hook_o" not in hook and "hook_z" not in hook:
                continue
            #print(f"hook: {hook}")

            head = int(parts[2].strip())
            cos_sim = float(parts[3].strip())
            rows.append((hook, head, cos_sim))
            if hook not in seen_hooks:
                seen_hooks.append(hook)
            seen_heads.add(head)

    if not rows:
        print("No data rows found.")
        return

    # ── build heatmap matrix ──────────────────────────────────────────
    hook_index_of = {h: i for i, h in enumerate(seen_hooks)}
    n_hooks = len(seen_hooks)
    sorted_heads = sorted(seen_heads)
    n_heads = len(sorted_heads)
    head_index_of = {h: i for i, h in enumerate(sorted_heads)}

    # NaN fill so missing combos show as a distinct color (via cmap.set_bad)
    matrix = np.full((n_heads, n_hooks), np.nan)
    for hook, head, cos_sim in rows:
        matrix[head_index_of[head], hook_index_of[hook]] = cos_sim

    # ── draw heatmap ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_hooks * 0.6), max(4, n_heads * 0.5)))

    cmap = plt.cm.RdYlGn  # red (low) → yellow → green (high)
    cmap.set_bad(color='lightgray')

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", rotation=270, labelpad=15)

    # Tick labels – abbreviated hook names
    index = 0
    for item in seen_hooks:
        seen_hooks[index] = item.replace("blocks.", "D")
        index += 1

    short_names = [
        h.split(".attn.")[0] + "." + h.split(".attn.")[1] if ".attn." in h else h
        for h in seen_hooks
    ]
    #print(short_names)
    ax.set_xticks(range(n_hooks))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"Head {h}" for h in sorted_heads], fontsize=8)

    ax.set_xlabel("Hook")
    ax.set_ylabel("Head")
    ax.set_title(f"Cosine similarity heatmap — {filename}")

    # Annotate cells with values if grid is not too large
    if n_hooks * n_heads <= 200:
        for i in range(n_heads):
            for j in range(n_hooks):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="black")

    fig.tight_layout()

    outpath = SNAPSHOT_DATA_PATH / f"{filename}_hook_o_z_only.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to {outpath}")


def plot_cosine_chart4(filename: str) -> None:
    """Read cosine similarity per-head CSV data and plot a line-segment chart.

    The CSV is expected at ``snapshots/data/<filename>`` with the format::

        hook1, hook2, head, cosine_similarity

    For each attention head a piecewise-linear path is drawn.  Each segment
    spans one unit on the x-axis and its *slope* is set to
    ``arccos(cosine_similarity)``.  Cosine similarity close to 1 therefore
    produces a flat segment, while a low similarity produces a steep upward
    segment.

    The output image is saved to ``snapshots/data/<filename>.png``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    filepath = SNAPSHOT_DATA_PATH / filename
    if not filepath.is_file():
        print(f"File not found: {filepath}")
        return

    # ── read CSV ──────────────────────────────────────────────────────
    rows: list[tuple[str, int, float]] = []     # (hook, head, cos_sim)
    seen_hooks: list[str] = []
    seen_heads: set[int] = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            hook = parts[0].strip()
            head = int(parts[2].strip())
            cos_sim = float(parts[3].strip())
            rows.append((hook, head, cos_sim))
            if hook not in seen_hooks:
                seen_hooks.append(hook)
            seen_heads.add(head)

    if not rows:
        print("No data rows found.")
        return

    # Build (hook_index, cos_sim) per head, preserving file order
    hook_index_of = {h: i for i, h in enumerate(seen_hooks)}
    heads: dict[int, list[tuple[int, float]]] = {h: [] for h in seen_heads}
    for hook, head, cos_sim in rows:
        heads[head].append((hook_index_of[hook], cos_sim))

    # print table of angles per head
    print()
    print(
        f"{'hook_name':<36}"
        f"{'head':>15}"
        f"{'angle':>13}"
    )

    for head in sorted(heads.items():
        for _hook_idx, cos_sim in sorted(segments):
            angle = math.acos(max(-1.0, min(1.0, cos_sim)))  # clamp to [-1, 1]
            print(
                f"{hookhook:<36} "
                f"{head:>15} "
                f"{angle:>12.4f} "
            )

    # ── draw ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    n_heads = len(heads)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_heads))
    
    for (head, segments), color in zip(
        sorted(heads.items()), colors
    ):
        x = 0.0
        y = 0.0
        xs: list[float] = [x]
        ys: list[float] = [y]
        for _hook_idx, cos_sim in sorted(segments):
            angle = math.acos(max(-1.0, min(1.0, cos_sim)))  # clamp to [-1, 1]
            x += 1.0
            y = angle
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, color=color, label=f"Head {head}")

    ax.set_xlabel("Hook index (each unit = one hook)")
    ax.set_ylabel("Cumulative arccos(cosine similarity)")
    ax.set_title(f"Cosine similarity change per head — {filename}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

    # Tick labels – abbreviated hook names
    index = 0
    for item in seen_hooks:
        seen_hooks[index] = item.replace("blocks.", "D")
        index += 1

    short_names = [
        #h.split(".attn.")[0] + "." + h.split(".attn.")[1][:8] if ".attn." in h else h
        h.split(".attn.")[0] + "." + h.split(".attn.")[1] if ".attn." in h else h
        for h in seen_hooks
    ]

    ax.set_xticks(range(len(seen_hooks)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)

    fig.tight_layout()

    outpath = SNAPSHOT_DATA_PATH / f"{filename}_4.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {outpath}")


