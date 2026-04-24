"""Model command handlers."""

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

    model_name = context.args[0]

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


