from tflens_explorer.cli.dispatcher import CommandDispatcher
from tflens_explorer.cli.command_registry import build_registry
from tflens_explorer.core.types import CommandContext
from tflens_explorer.core.session import AppSession
from tflens_explorer.handlers.prompt_handlers import parse_kv_args


def test_clear_session_clear_prompt_and_outputs():
    registry = build_registry()
    session = AppSession()
    session.current_prompt = "Hello"
    session.last_output = "Something"
    session.logits = object()
    session.tokens = object()
    session.cache = object()

    dispatcher = CommandDispatcher(registry, session)

    dispatcher.dispatch("session-clear")

    assert session.current_prompt == ""
    assert session.last_output == ""
    assert session.logits is None
    assert session.tokens is None
    assert session.cache is None

def test_clear_dispatches_without_crashing():
    registry = build_registry()
    session = AppSession()
    dispatcher = CommandDispatcher(registry, session)

    dispatcher.dispatch("clear")

def test_context_show():
    registry = build_registry()
    session = AppSession()
    session.current_prompt = "Hello"
    session.last_output = "Something"
    session.logits = object()
    session.tokens = object()
    session.cache = object()

    dispatcher = CommandDispatcher(registry, session)

    dispatcher.dispatch("context-show")

    assert session.current_prompt == "Hello"
    assert session.last_output == "Something"
    assert session.logits is not None
    assert session.tokens is not None
    assert session.cache is not None

    
