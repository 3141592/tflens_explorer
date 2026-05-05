import io
import sys
from tflens_explorer.cli.dispatcher import CommandDispatcher
from tflens_explorer.cli.command_registry import build_registry
from tflens_explorer.core.types import CommandContext
from tflens_explorer.core.session import AppSession
from tests.fakes import FakeModel


def test_cache_keys_dispatches_to_handler():
    registry = build_registry()
    session = AppSession()
    dispatcher = CommandDispatcher(registry, session)

    captured = {}

    def fake_handler(ctx):
        captured["called"] = True

    registry.get("cache-keys").handler = fake_handler

    dispatcher.dispatch("cache-keys")

    assert captured["called"] is True

def test_cache_run_dispatches_to_handler():
    registry = build_registry()
    session = AppSession()
    dispatcher = CommandDispatcher(registry, session)

    captured = {}

    def fake_handler(ctx):
        captured["called"] = True

    registry.get("cache-run").handler = fake_handler

    dispatcher.dispatch("cache-run")

    assert captured["called"] is True

def test_cache_show_dispatches_to_handler():
    registry = build_registry()
    session = AppSession()
    dispatcher = CommandDispatcher(registry, session)

    captured = {}

    def fake_handler(ctx):
        captured["called"] = True

    registry.get("cache-show").handler = fake_handler

    dispatcher.dispatch("cache-show")

    assert captured["called"] is True

def test_cache_keys_outputs_keys():
    registry = build_registry()
    session = AppSession()
    session.model = FakeModel()
    session.current_prompt = "Hello"

    session.cache = {
        ("pattern", 0): "value1",
        ("pattern", 1): "value2",
    }

    dispatcher = CommandDispatcher(registry, session)

    captured = io.StringIO()
    sys.stdout = captured

    dispatcher.dispatch("cache-keys")

    sys.stdout = sys.__stdout__

    output = captured.getvalue()
    assert "pattern" in output

def test_cache_show_outputs_keys():
    registry = build_registry()
    session = AppSession()
    session.model = FakeModel()
    session.current_prompt = "Hello"

    session.tokens = [1, 2, 3]
    session.logits = [[0.1, 0.2]]
    session.current_prompt = "Hello"
    session.cache = {
        ("pattern", 0): "value1",
        ("pattern", 1): "value2",
    }

    dispatcher = CommandDispatcher(registry, session)

    captured = io.StringIO()
    sys.stdout = captured

    dispatcher.dispatch("cache-show")

    sys.stdout = sys.__stdout__

    output = captured.getvalue()
    assert "Hello" in output
    assert "cache_keys" in output


