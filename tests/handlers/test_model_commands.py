import torch
import pytest
from types import SimpleNamespace
from unittest.mock import patch
from tflens_explorer.cli.dispatcher import CommandDispatcher
from tflens_explorer.cli.command_registry import build_registry
from tflens_explorer.core.session import AppSession
from tflens_explorer.core.types import CommandContext
from tflens_explorer.handlers.model_handlers import handle_model_load
from tflens_explorer.handlers.model_handlers import handle_model_info
from tflens_explorer.services.model_service import get_model_info
from tflens_explorer.handlers.model_handlers import handle_model_cache

@patch("tflens_explorer.services.model_service.load_model")
def test_handle_model_load_updates_session(mock_load_model):
    fake_model = object()
    mock_load_model.return_value = fake_model

    session = AppSession()
    registry = build_registry()
    context = CommandContext(
        raw='model-load openai-community/gpt2',
        args=["openai-community/gpt2"],
        session=session,
        registry=registry,
    )

    handle_model_load(context)

    mock_load_model.assert_called_once_with("openai-community/gpt2")
    assert session.model is fake_model
    assert session.current_model_name == "openai-community/gpt2"


@patch("tflens_explorer.services.model_service.load_model")
def test_handle_model_load_without_args(mock_load_model, capsys):
    fake_model = object()
    mock_load_model.return_value = fake_model

    session = AppSession()
    registry = build_registry()
    context = CommandContext(
        raw='model-load openai-community/gpt2',
        args=[],
        session=session,
        registry=registry,
    )

    handle_model_load(context)

    mock_load_model.assert_not_called()
    assert session.model is None

    captured = capsys.readouterr()
    assert "Usage" in captured.out 


@patch("tflens_explorer.services.model_service.load_model")
def test_handle_model_load_handles_load_failure(
    mock_load_model,
    capsys,
):
    mock_load_model.side_effect = RuntimeError("Failed to load model")

    session = AppSession()
    registry = build_registry()

    context = CommandContext(
        raw="model-load gpt2",
        args=["gpt2"],
        session=session,
        registry=registry,
    )

    handle_model_load(context)

    assert session.model is None

    captured = capsys.readouterr()
    assert "Failed to load model" in captured.out


@patch("tflens_explorer.services.model_service.get_model_info")
def test_handle_model_info_prints_model_info(
    mock_get_model_info,
    capsys,
):
    session = AppSession()
    session.model = object()

    registry = build_registry()

    mock_get_model_info.return_value = {
        "n_layers": 12,
        "d_model": 768,
    }

    context = CommandContext(
        raw="model-info",
        args=[],
        session=session,
        registry=registry,
    )

    handle_model_info(context)

    mock_get_model_info.assert_called_once_with(session.model)

    captured = capsys.readouterr()

    assert "n_layers" in captured.out
    assert "12" in captured.out


@patch("tflens_explorer.services.model_service.get_model_info")
def test_handle_model_info_without_loaded_model(mock_get_model_info, capsys):
    session = AppSession()
    registry = build_registry()

    context = CommandContext(
        raw="model-info",
        args=[],
        session=session,
        registry=registry,
    )

    handle_model_info(context)

    mock_get_model_info.assert_not_called()

    captured = capsys.readouterr()
    assert "No model loaded." in captured.out


def test_handle_model_cache(capsys):
    session = AppSession()
    registry = build_registry()

    context = CommandContext(
        raw="model-cache",
        args=[],
        session=session,
        registry=registry,
    )

    handle_model_cache(context)

    captured = capsys.readouterr()
    assert "Cached Hugging Face models:" in captured.out

