"""CLI entry point."""

from huggingface_hub import login
import os

from tflens_explorer.cli.dispatcher import CommandDispatcher
from tflens_explorer.cli.command_registry import build_registry
from tflens_explorer.core.session import AppSession


def main() -> int:
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)

    session = AppSession()
    registry = build_registry()
    dispatcher = CommandDispatcher(registry=registry, session=session)

    print("tflens-explorer")
    print("Type 'help' for commands. Type 'quit' to exit.")

    while session.running:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        dispatcher.dispatch(raw)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
