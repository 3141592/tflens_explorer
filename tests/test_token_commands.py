import io
import sys
from tflens_explorer.cli.dispatcher import CommandDispatcher
from tflens_explorer.cli.command_registry import build_registry
from tflens_explorer.core.types import CommandContext
from tflens_explorer.core.session import AppSession
from tests.fakes import FakeModel

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
    session.model = FakeModel()
    dispatcher = CommandDispatcher(registry, session)

    captured_output = io.StringIO()
    sys.stdout = captured_output

    dispatcher.dispatch("token-decode 15496")
    sys.stdout = sys.__stdout__

    assert "Hello" in captured_output.getvalue()


def test_token_encode():
    registry = build_registry()
    session = AppSession()
    session.model = FakeModel()
    dispatcher = CommandDispatcher(registry, session)

    captured_output = io.StringIO()
    sys.stdout = captured_output

    dispatcher.dispatch("token-encode Hello")
    sys.stdout = sys.__stdout__

    assert "5" in captured_output.getvalue()

