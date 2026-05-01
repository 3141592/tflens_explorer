import io
import sys
from types import SimpleNamespace
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

class FakeTokenizer:
    def encode(self, text):
        return [len(text)]  # simple, deterministic

    def decode(self, token_ids):
        return "Hello"

class FakeModel:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.cfg = SimpleNamespace(d_vocab=50257)

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

