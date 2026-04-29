from tflens_explorer.cli.dispatcher import CommandDispatcher
from tflens_explorer.cli.command_registry import build_registry
from tflens_explorer.core.types import CommandContext
from tflens_explorer.core.session import AppSession


def test_token_decode_dispatches_to_handler():
    registry = build_registry()
    session = AppSession()
    dispatcher = CommandDispatcher(registry, session)

    captured = {}

    def fake_handler(ctx):
        captured["called"] = True

    registry.get("token-decode").handler = fake_handler

    dispatcher.dispatch("token-decode 15496")

    assert captured["called"] is True

def test_token_encode_dispatches_to_handler():
    registry = build_registry()
    session = AppSession()
    dispatcher = CommandDispatcher(registry, session)

    captured = {}

    def fake_handler(ctx):
        captured["called"] = True

    registry.get("token-encode").handler = fake_handler

    dispatcher.dispatch("token-encode Dog")

    assert captured["called"] is True

def test_token_decode():
    registry = build_registry()
    session = AppSession()
    dispatcher = CommandDispatcher(registry, session)

    captured = {}

    def fake_handler(ctx):
        captured["input"] = ctx.raw

    registry.get("token-decode").handler = fake_handler

    dispatcher.dispatch("token-decode 10057")

    assert "10057" in captured["input"]

def test_token_encode():
    registry = build_registry()
    session = AppSession()
    dispatcher = CommandDispatcher(registry, session)

    captured = {}

    def fake_handler(ctx):
        captured["input"] = ctx.raw

    registry.get("token-encode").handler = fake_handler

    dispatcher.dispatch("token-encode system")

    assert "system" in captured["input"]

