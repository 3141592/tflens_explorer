"""Compare Service"""

import os
import yaml
import torch
import re
import traceback
import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tflens_explorer.services.model_service import tokens_for_snapshot, logits_for_snapshot
from tflens_explorer.services.model_service import cache_summary_for_snapshot, get_model_alias
from tflens_explorer.services.model_service import cache_summary_for_snapshot_all
from tflens_explorer.core.types import CommandContext
from tflens_explorer.cli.utilities import get_shape
from tflens_explorer.core.snapshot_types import Snapshot, SNAPSHOT_PATH
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
        tokens=tokens_for_snapshot(current_model, prompt, prepend_bos),
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

def compare_snapshots(snapshot1: Snapshot, snapshot2: Snapshot):
    all_args = locals()
    for name, value in all_args.items():
        if verify_snapshot(value):
            pass
        else:
            print(f"Snapshot {value} does not exist. Use: snapshots-list to find valid snapshots.")
            return

    snapshot1 = Snapshot.load(snapshot1)
    snapshot2 = Snapshot.load(snapshot2)

    token_size_comparison = snapshot1.tokens[0] == snapshot2.tokens[0]
    token_id_comparison = compare_token_ids(snapshot1.tokens, snapshot2.tokens)
    token_comparison = compare_tokens(snapshot1.tokens, snapshot2.tokens)
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
    print(f"    A: {str(logits_1_size)}")
    print(f"    B: {str(logits_2_size)}")
    print(f"  top-1:")
    print(f"    A: {logit_comparison[0][0]}")
    print(f"    B: {logit_comparison[0][1]}")
    print(f"  top-5 overlap: {logit_comparison[1]}/5")
    print()
    if len(snapshot1.cache) > 0 and len(snapshot2.cache) > 0:
        print(f"Cache activation differences:")
        cache_activation_summary(snapshot1.cache, snapshot2.cache)
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
        elif item['token_id'] == tokens2[index]['token_id']:
            token_id_comparison[index] = True
        else:
            token_id_comparison[index] = False
    for index, item in enumerate(tokens2):
        try:
            if len(tokens1) <= index:
                token_id_comparison[index] = False
            elif item['token_id'] == tokens1[index]['token_id']:
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
        elif item['token'] == tokens2[index]['token']:
            token_comparison[index] = True
        else:
            token_comparison[index] = False

    for index, item in enumerate(tokens2):
        try:
            if len(tokens1) <= index:
                token_comparison[index] = False
            elif item['token'] == tokens1[index]['token']:
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

def cache_activation_summary(cache1, cache2):
    print(
        f"    {'A/B':<4}"
        f"{'hook_name':<36}"
        f"{'shape':>15}"
        f"{'min':>13}"
        f"{'max':>13}"
        f"{'mean':>13}"
    )

    different_values_count = 0
    for activation1, activation2 in zip(cache1, cache2):
        hook1 = activation1.hook
        hook2 = activation2.hook
        shape1 = get_shape(activation1.shape)
        shape2 = get_shape(activation2.shape)
        minimum1 = activation1.minimum
        minimum2 = activation2.minimum
        maximum1 = activation1.maximum
        maximum2 = activation2.maximum
        
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
            print(f"Cache hooks {hook1} and {hook2} do not match.")
            return

        if (minimum1 == minimum2) and (maximum1 == maximum2):
            continue
        else:
            different_values_count += 1

        print(
            f"    {'A:':<4}"
            f"{hook1:<35} "
            f"{shape1:>15} "
            f"{minimum1:>12.4f} "
            f"{maximum1:>12.4f} "
            f"{mean1_str:>12}"
        )
                
        print(
            f"    {'B:':<4}"
            f"{hook2:<35} "
            f"{shape2:>15} "
            f"{minimum2:>12.4f} "
            f"{maximum2:>12.4f} "
            f"{mean2_str:>12}"
        )
        
        print()

    if different_values_count == 0:
        print("    No cache summary differences found.")

    print()
    return

# TODO:
# Reintroduce advanced tensor comparison once optional
# raw tensor snapshot storage exists.
def cache_activation_summary_2(cache1, cache2):
    print(f"    {'hook_name':<35} {'shape':<15} {'mean_abs_diff':>15} {'max_abs_delta':>15} {'cosine_sim':>15}")

    different_values_count = 0
    for activation1, activation2 in zip(cache1, cache2):
        hook1 = activation1.layer
        hook2 = activation2.layer

        if hook1 != hook2:
            print(f"Cache hooks {hook1} and {hook2} do not match.")
            return

        value1 = torch.tensor(activation1["value"], dtype=torch.float32).flatten()
        value2 = torch.tensor(activation2["value"], dtype=torch.float32).flatten()

        if value1.shape != value2.shape:
            print(f"Shape mismatch for {hook1}: {value1.shape} vs {value2.shape}")
            continue

        diff = value1 - value2
        abs_diff = diff.abs()

        mean_abs_diff = round(abs_diff.mean().item(), 4)
        max_abs_delta = round(abs_diff.max().item(), 4)

        cosine_sim = torch.nn.functional.cosine_similarity(
            value1.unsqueeze(0),
            value2.unsqueeze(0),
            dim=1,
        ).item()

        cosine_sim = round(cosine_sim, 4)

        shape = get_shape(activation1['shape'])
        
        if (mean_abs_diff == 0.0) and (max_abs_delta == 0.0) and ((cosine_sim == 0.0) or (cosine_sim == 1.0)):
            continue
        else:
            different_values_count += 1

        print(
            f"    {hook1:<35} "
            f"{shape:<15} "
            f"{mean_abs_diff:>15} "
            f"{max_abs_delta:>15} "
            f"{cosine_sim:>15}"
        )

    if different_values_count == 0:
        print("    All A: and B: values are the same.")

    print()

    return

def snapshots_have_raw_cache_values(cache1, cache2):
    if not cache1 or not cache2:
        return False
