"""Model command handlers."""


import json
from pathlib import Path

from tflens_explorer.core.types import CommandContext


def handle_model_list(context: CommandContext) -> None:
    from tflens_explorer.services.model_service import list_models
    
    arg = ""
    if context.args:
        arg = context.args[0]

    models = list_models(arg)
    for item in models:
        print(item)


def handle_model_load(context: CommandContext) -> None:
    if not context.args:
        print("Usage: model-load <model_name>")
        return

    from tflens_explorer.services.model_service import resolve_model_name

    input_name = context.args[0]
    model_name = resolve_model_name(input_name)

    if model_name != input_name:
        print(f"Resolved alias: {input_name} -> {model_name}")

    from tflens_explorer.services.model_service import load_model

    try:
        model = load_model(model_name)
    except Exception as exc:
        print(f"Failed to load model '{model_name}': {exc}")
        return

    context.session.model = model
    context.session.current_model_name = model_name
    context.session.cache = None

    print()
    print(f"Loaded model: {model_name}")


def handle_model_info(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    from tflens_explorer.services.model_service import get_model_info

    info = get_model_info(model)

    for key, value in info.items():
        print(f"{key}: {value}")


from tflens_explorer.config.config_loader import load_model_aliases
def handle_model_aliases(context) -> None:
    aliases = load_model_aliases()

    if not aliases:
        print("No model aliases configured.")
        return

    print("Model aliases:")
    print()

    for alias, model_id in sorted(aliases.items()):
        print(f"{alias:<20} -> {model_id}")


def handle_model_cache(context) -> None:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_dir.exists():
        print("Hugging Face cache directory not found.")
        return

    model_dirs = sorted(cache_dir.glob("models--*"))

    if not model_dirs:
        print("No cached models found.")
        return

    print("Cached Hugging Face models:\n")
    print(f"{'model':<45} {'kind':<10} detail")
    print("-" * 70)
    for model_dir in model_dirs:
        model_id = model_dir.name.replace("models--", "").replace("--", "/")

        snapshots_dir = model_dir / "snapshots"
        snapshot_dirs = sorted(snapshots_dir.iterdir()) if snapshots_dir.exists() else []

        if not snapshot_dirs:
            #print(f"{model_id:<45} unknown  no snapshots")
            status = "unknown"
            detail = "no snapshots"
            print(f"{model_id:<45} {status:<10} {detail}")
            continue

        latest_snapshot = snapshot_dirs[-1]
        config_path = latest_snapshot / "config.json"

        if not config_path.exists():
            status = "invalid"
            detail = "no config.json"
            print(f"{model_id:<45} {status:<10} {detail}")
            continue

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                model_type = config.get("model_type")
                archs = config.get("architectures", [])
        except json.JSONDecodeError:
            status = "invalid"
            detail = "invalid config.json"
            print(f"{model_id:<45} {status:<10} {detail}")
            continue

        if any(("CausalLM" in arch or "LMHeadModel" in arch) for arch in archs):
            status = "LLM"
        elif model_type:
            status = "non-LLM"
        else:
            status = "no"

        if "model_type" in config:
            detail = f"model_type={model_type}"
        else:
            detail = "missing model_type"

        print(f"{model_id:<45} {status:<10} {detail}")