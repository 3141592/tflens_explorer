from tflens_explorer.cli.dispatcher import CommandDispatcher
from tflens_explorer.cli.command_registry import build_registry
from tflens_explorer.core.types import CommandContext
from tflens_explorer.core.session import AppSession


def test_prompt_set_updates_context():
    registry = build_registry()
    session = AppSession()
    context = CommandContext('', '', session, registry)
    dispatcher = CommandDispatcher(registry, context.session)
    dispatcher.dispatch("prompt-set Hello")
    assert context.session.current_prompt == "Hello"

def test_unknown_command_does_not_crash():
    registry = build_registry()
    session = AppSession()
    context = CommandContext("", "", session, registry)
    dispatcher = CommandDispatcher(registry, context.session)

    dispatcher.dispatch("not-a-real-command")

    # assert something stable about behavior
    # for example, session unchanged:
    assert context.session.current_prompt == ""

def test_dispatch_routes_to_prompt_run():
    registry = build_registry()
    session = AppSession()
    context = CommandContext("", "", session, registry)
    dispatcher = CommandDispatcher(registry, context.session)

    captured = {}

    def fake_handler(ctx):
        captured["called"] = True
        captured["kwargs"] = getattr(ctx, "kwargs", {})

    # Replace the real handler
    command = registry.get("prompt-run")
    command.handler = fake_handler
    dispatcher.dispatch("prompt-run new_tokens=25")

    assert captured.get("called") is True
