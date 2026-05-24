"""Compare Service"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tflens_explorer.services.model_service import tokens_for_snapshot, logits_for_snapshot, cache_summary_for_snapshot, get_model_alias
from tflens_explorer.core.types import CommandContext

base_dir = Path(__file__).resolve().parents[3]
snapshot_path = base_dir / "snapshots"
snapshot_path.mkdir(parents=True, exist_ok=True)

@dataclass
class Model:
    name: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    num_ctx: int | None = None
    prepend_bos: bool | None = None
    layers: int | None = None
    heads: int | None = None
    vocabulary: int | None = None    

    def save(self) -> None:
        path = snapshot_path / f"{self.name}.yaml"
        serial = dict(self.__dict__)                    # shallow copy
        # convert model to plain dict (choose attributes you need)
        serial["model"] = {
            "name": self.model.name,
            "temperature": self.model.temperature,
            "top_k": self.model.top_k,
            "top_p": self.model.top_p,
            "num_ctx": self.model.num_ctx,
            "prepend_bos": self.model.prepend_bos,
            "layers": self.model.layers,
            "heads": self.model.heads,
            "vocabulary": self.model.vocabulary,
        }
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(serial, f, sort_keys=False)

@dataclass
class Snapshot:
    name: str | None = None
    model: Model | None = None
    prompt: str | None = None
    tokens: list[int] = field(default_factory=list)
    logits: list[float] = field(default_factory=list)
    cache: list[int] = field(default_factory=list)

    def save(self) -> None:
        """Save the snapshot to a YAML file."""
        try:
            path = snapshot_path / f"{self.name}.yaml"

            # Convert Model object
            serial = dict(self.__dict__)
            serial["model"] = asdict(self.model)

            with path.open("w", encoding="utf-8") as f:
                # dump a plain dict instead of the object to avoid yaml picking class internals
                yaml.safe_dump(serial, f, sort_keys=False)
            print(f"Snapshot saved to {path}")
        except Exception as e:
            print(f"Error saving snapshot: {str(e)}")
            raise

    def load(self) -> None:
        """Load the snapshot to a Snapshot object."""
        try:
            path = snapshot_path / f"{self.name}.yaml"
            import yaml

            with open(path, 'r') as file:
                data = yaml.safe_load(file)

            self.name = data['name']
            self.prompt = data['prompt']
            # Model
            self.model = Model()
            self.model.name = data['model']['name']
            self.model.temperature = data['model']['temperature']
            self.model.top_k = data['model']['top_k']
            self.model.top_p = data['model']['top_p']
            self.model.num_ctx = data['model']['num_ctx']
            self.model.prepend_bos = data['model']['prepend_bos']
            self.model.layers = data['model']['layers']
            self.model.heads = data['model']['heads']
            self.model.vocabulary = data['model']['vocabulary']
            
            # Tokens
            self.tokens = []
            for item in data['tokens']:
                self.tokens.append(item)

            # Logits
            self.logits = []
            for item in data['logits']:
                self.logits.append(item)

            # Cache
            self.cache = []
            for item in data['cache']:
                self.cache.append({item: data['cache'][item]})

        except Exception as error:
            print(f"Exception: {error}")
            

class Compare:
    def __init__(self, name: str) -> None:
        self.snapshot = Snapshot(name=name)
        # Additional initialization could be added here

def snapshot_create(context: CommandContext, snapshot_name: str, layer: str) -> None:
    """Create a model comparison snapshot."""
    model_name = context.session.current_model_name
    current_model = context.session.model
    prompt = context.session.current_prompt
    prepend_bos = context.session.prepend_bos
    model_alias = get_model_alias(model_name)

    if len(model_alias) > 0:
        model_name = f"{model_alias} -> {model_name}"

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
        cache = cache_summary_for_snapshot(current_model, prompt, layer)
    except:
        print(f"Layer name {layer} is not valid.")
        return

    snapshot = Snapshot(
        name=snapshot_name,
        model=model,
        prompt=prompt,
        tokens=tokens_for_snapshot(current_model, prompt, prepend_bos),
        logits=logits_for_snapshot(current_model, prompt, prepend_bos),
        cache=cache,
    )
    
    snapshot.save()

def snapshots_list():
    p = Path(snapshot_path)
    for f in p.iterdir():
        if f.is_file():
            print(f.stem)

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

def compare_cache():
    print("compare-cache")

def verify_snapshot(snapshot_name):
    p = Path(f"{snapshot_path}/{snapshot_name}.yaml")
    if p.exists():
        return True
    else:
        return False

def compare_snapshots(snapshot1: Snapshot, snapshot2: Snapshot):
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
            breakpoint()
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
            breakpoint()

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
        print(f"    No rankings in common.")
    
    print()

    return