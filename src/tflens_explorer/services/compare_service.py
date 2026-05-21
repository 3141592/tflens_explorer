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

def compare_tokens():
    print("compare-tokens")

def compare_logits():
    print("compare-logits")

def compare_cache():
    print("compare-cache")

def verify_snapshot(snapshot_name):
    p = Path(f"{snapshot_path}/{snapshot_name}.yaml")
    if p.exists():
        return True
    else:
        return False

def compare_models(snapshot1: Snapshot, snapshot2: Snapshot):
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

    snapshot1_token_list = get_token_list(snapshot1.tokens)
    snapshot2_token_list = get_token_list(snapshot2.tokens)

    print(f"Models:")
    print(f"  A: {snapshot1.model.name}")
    print(f"  B: {snapshot2.model.name}")
    print()
    print(f"Prompt:")
    print(f"  A: {snapshot1.prompt}")
    print(f"  B: {snapshot2.prompt}")
    print()
    print(f"Tokenization:")
    print(f"  A: {snapshot1.tokens}")
    print(f"  B: {snapshot2.tokens}")
    print()

def get_token_list(tokens):
    for item in tokens:
        breakpoint()
    return []