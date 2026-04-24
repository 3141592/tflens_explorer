"""Model command handlers."""

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

    for model_dir in model_dirs:
        # models--openai-community--gpt2 → openai-community/gpt2
        name = model_dir.name.replace("models--", "").replace("--", "/")
        print(name)