from tflens_explorer.cli.dispatcher import CommandDispatcher
from tflens_explorer.cli.command_registry import build_registry
from tflens_explorer.core.types import CommandContext
from tflens_explorer.core.session import AppSession
from tflens_explorer.handlers.prompt_handlers import parse_kv_args


def test_parse_kv_args_extracts_new_tokens():
    args = parse_kv_args(["new_tokens=25"])
    assert args == {"new_tokens": 25}

def test_parse_kv_args_handles_empty_args():
    args = parse_kv_args([])
    assert args == {}

def test_parse_kv_args_ignores_invalid_pairs():
    args = parse_kv_args(["badinput"])
    assert args == {}

def test_prompt_set_updates_context():
    registry = build_registry()
    session = AppSession()
    context = CommandContext('', '', session, registry)
    dispatcher = CommandDispatcher(registry, context.session)
    dispatcher.dispatch("prompt-set Hello")
    assert context.session.current_prompt == "Hello"

def test_prompt_show_empty():
    registry = build_registry()
    session = AppSession()
    context = CommandContext('', '', session, registry)
    dispatcher = CommandDispatcher(registry, context.session)
    assert context.session.current_prompt == ''

def test_prompt_show_not_empty():
    registry = build_registry()
    session = AppSession()
    context = CommandContext('', '', session, registry)
    dispatcher = CommandDispatcher(registry, context.session)
    dispatcher.dispatch("prompt-set This is a new prompt")
    assert context.session.current_prompt == 'This is a new prompt'

