"""Tests for compare_handlers.py."""
import pytest
from unittest.mock import patch
from tflens_explorer.cli.command_registry import build_registry
from tflens_explorer.core.session import AppSession
from tflens_explorer.core.types import CommandContext
from tflens_explorer.handlers.compare_handlers import (
    handle_snapshot_create,
    handle_snapshots_list,
    handle_compare_models,
    handle_compare_attention,
    handle_compare_mlp,
    handle_compare_generated,
    handle_compare_evals,
    handle_compare_tokens,
    handle_compare_logits,
    handle_compare_cache,
    handle_compare_snapshots,
    linux_filename_validation,
)


# ---------------------------------------------------------------------------
# linux_filename_validation
# ---------------------------------------------------------------------------

class TestLinuxFilenameValidation:
    def test_valid_name(self):
        assert linux_filename_validation("my_snapshot") is True
        assert linux_filename_validation("snapshot-1") is True
        assert linux_filename_validation("a" * 255) is True

    def test_contains_slash(self):
        assert linux_filename_validation("foo/bar") is False

    def test_contains_null(self):
        assert linux_filename_validation("foo\0bar") is False

    def test_too_long(self):
        assert linux_filename_validation("a" * 256) is False


# ---------------------------------------------------------------------------
# handle_snapshot_create
# ---------------------------------------------------------------------------

class TestHandleSnapshotCreate:
    def make_context(self, session, args):
        registry = build_registry()
        return CommandContext(
            raw="snapshot-create",
            args=args,
            session=session,
            registry=registry,
        )

    def test_no_model_loaded(self, capsys):
        session = AppSession()
        context = self.make_context(session, [])
        handle_snapshot_create(context)
        captured = capsys.readouterr()
        assert "No model loaded." in captured.out

    def test_no_prompt_set(self, capsys):
        session = AppSession()
        session.model = object()
        context = self.make_context(session, [])
        handle_snapshot_create(context)
        captured = capsys.readouterr()
        assert "No prompt set." in captured.out

    def test_no_snapshot_name(self, capsys):
        session = AppSession()
        session.model = object()
        session.current_prompt = "hello"
        context = self.make_context(session, ["hook=blocks.0.hook_resid_pre"])
        handle_snapshot_create(context)
        captured = capsys.readouterr()
        assert "A snapshot name is required" in captured.out

    def test_invalid_snapshot_name(self, capsys):
        session = AppSession()
        session.model = object()
        session.current_prompt = "hello"
        context = self.make_context(session, ["name=foo/bar", "hook=blocks.0.hook_resid_pre"])
        handle_snapshot_create(context)
        captured = capsys.readouterr()
        assert "valid Linux filename" in captured.out

    def test_no_hook(self, capsys):
        session = AppSession()
        session.model = object()
        session.current_prompt = "hello"
        context = self.make_context(session, ["name=my_snapshot"])
        handle_snapshot_create(context)
        captured = capsys.readouterr()
        assert "A cache hook name is required" in captured.out

    @patch("tflens_explorer.handlers.compare_handlers.snapshot_create")
    def test_valid_call(self, mock_snapshot_create, capsys):
        session = AppSession()
        session.model = object()
        session.current_prompt = "hello"
        context = self.make_context(session, ["name=my_snapshot", "hook=blocks.0.hook_resid_pre"])
        handle_snapshot_create(context)
        mock_snapshot_create.assert_called_once_with(context, "my_snapshot", "blocks.0.hook_resid_pre")
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# handle_snapshots_list
# ---------------------------------------------------------------------------

class TestHandleSnapshotsList:
    @patch("tflens_explorer.handlers.compare_handlers.snapshots_list")
    def test_delegates(self, mock_snapshots_list):
        session = AppSession()
        registry = build_registry()
        context = CommandContext(
            raw="snapshots-list",
            args=[],
            session=session,
            registry=registry,
        )
        handle_snapshots_list(context)
        mock_snapshots_list.assert_called_once_with()


# ---------------------------------------------------------------------------
# handle_compare_models
# ---------------------------------------------------------------------------

class TestHandleCompareModels:
    @patch("tflens_explorer.handlers.compare_handlers.compare_models")
    def test_delegates(self, mock_compare_models):
        session = AppSession()
        registry = build_registry()
        context = CommandContext(
            raw="compare-models",
            args=[],
            session=session,
            registry=registry,
        )
        handle_compare_models(context)
        mock_compare_models.assert_called_once_with()


# ---------------------------------------------------------------------------
# handle_compare_attention
# ---------------------------------------------------------------------------

class TestHandleCompareAttention:
    @patch("tflens_explorer.handlers.compare_handlers.compare_attention")
    def test_delegates(self, mock_compare_attention):
        session = AppSession()
        registry = build_registry()
        context = CommandContext(
            raw="compare-attention",
            args=[],
            session=session,
            registry=registry,
        )
        handle_compare_attention(context)
        mock_compare_attention.assert_called_once_with()


# ---------------------------------------------------------------------------
# handle_compare_mlp
# ---------------------------------------------------------------------------

class TestHandleCompareMlp:
    @patch("tflens_explorer.handlers.compare_handlers.compare_mlp")
    def test_delegates(self, mock_compare_mlp):
        session = AppSession()
        registry = build_registry()
        context = CommandContext(
            raw="compare-mlp",
            args=[],
            session=session,
            registry=registry,
        )
        handle_compare_mlp(context)
        mock_compare_mlp.assert_called_once_with()


# ---------------------------------------------------------------------------
# handle_compare_generated
# ---------------------------------------------------------------------------

class TestHandleCompareGenerated:
    @patch("tflens_explorer.handlers.compare_handlers.compare_generated")
    def test_delegates(self, mock_compare_generated):
        session = AppSession()
        registry = build_registry()
        context = CommandContext(
            raw="compare-generated",
            args=[],
            session=session,
            registry=registry,
        )
        handle_compare_generated(context)
        mock_compare_generated.assert_called_once_with()


# ---------------------------------------------------------------------------
# handle_compare_evals
# ---------------------------------------------------------------------------

class TestHandleCompareEvals:
    @patch("tflens_explorer.handlers.compare_handlers.compare_evals")
    def test_delegates(self, mock_compare_evals):
        session = AppSession()
        registry = build_registry()
        context = CommandContext(
            raw="compare-evals",
            args=[],
            session=session,
            registry=registry,
        )
        handle_compare_evals(context)
        mock_compare_evals.assert_called_once_with()


# ---------------------------------------------------------------------------
# handle_compare_tokens
# ---------------------------------------------------------------------------

class TestHandleCompareTokens:
    @patch("tflens_explorer.handlers.compare_handlers.compare_tokens")
    def test_delegates(self, mock_compare_tokens):
        session = AppSession()
        registry = build_registry()
        context = CommandContext(
            raw="compare-tokens",
            args=[],
            session=session,
            registry=registry,
        )
        handle_compare_tokens(context)
        mock_compare_tokens.assert_called_once_with()


# ---------------------------------------------------------------------------
# handle_compare_logits
# ---------------------------------------------------------------------------

class TestHandleCompareLogits:
    def make_context(self, session, args):
        registry = build_registry()
        return CommandContext(
            raw="compare-logits",
            args=args,
            session=session,
            registry=registry,
        )

    def test_no_args(self, capsys):
        session = AppSession()
        context = self.make_context(session, None)
        handle_compare_logits(context)
        captured = capsys.readouterr()
        assert "Two snapshot names are required" in captured.out

    def test_wrong_number_of_args(self, capsys):
        session = AppSession()
        context = self.make_context(session, ["snap1"])
        handle_compare_logits(context)
        captured = capsys.readouterr()
        assert "Two snapshot names are required" in captured.out

    @patch("tflens_explorer.handlers.compare_handlers.compare_logits")
    def test_valid_args(self, mock_compare_logits):
        session = AppSession()
        context = self.make_context(session, ["snap1", "snap2"])
        handle_compare_logits(context)
        mock_compare_logits.assert_called_once_with("snap1", "snap2")


# ---------------------------------------------------------------------------
# handle_compare_cache
# ---------------------------------------------------------------------------

class TestHandleCompareCache:
    def make_context(self, session, args):
        registry = build_registry()
        return CommandContext(
            raw="compare-cache",
            args=args,
            session=session,
            registry=registry,
        )

    def test_no_args(self, capsys):
        session = AppSession()
        context = self.make_context(session, None)
        handle_compare_cache(context)
        captured = capsys.readouterr()
        assert "Two snapshot names are required" in captured.out

    def test_wrong_number_of_args(self, capsys):
        session = AppSession()
        context = self.make_context(session, ["snap1"])
        handle_compare_cache(context)
        captured = capsys.readouterr()
        assert "Two snapshot names are required" in captured.out

    @patch("tflens_explorer.handlers.compare_handlers.compare_cache")
    def test_valid_args(self, mock_compare_cache):
        session = AppSession()
        context = self.make_context(session, ["snap1", "snap2"])
        handle_compare_cache(context)
        mock_compare_cache.assert_called_once_with("snap1", "snap2")


# ---------------------------------------------------------------------------
# handle_compare_snapshots
# ---------------------------------------------------------------------------

class TestHandleCompareSnapshots:
    def make_context(self, session, args):
        registry = build_registry()
        return CommandContext(
            raw="compare-snapshots",
            args=args,
            session=session,
            registry=registry,
        )

    def test_less_than_two_args(self, capsys):
        session = AppSession()
        context = self.make_context(session, ["snap1"])
        handle_compare_snapshots(context)
        captured = capsys.readouterr()
        assert "Two snapshot names are required" in captured.out

    @patch("tflens_explorer.handlers.compare_handlers.compare_snapshots")
    def test_two_args_no_kwargs(self, mock_compare_snapshots):
        session = AppSession()
        context = self.make_context(session, ["snap1", "snap2"])
        handle_compare_snapshots(context)
        mock_compare_snapshots.assert_called_once_with("snap1", "snap2", None, None)

    @patch("tflens_explorer.handlers.compare_handlers.compare_snapshots")
    def test_with_diff_kwarg(self, mock_compare_snapshots):
        session = AppSession()
        context = self.make_context(session, ["snap1", "snap2", "diff=0.5"])
        handle_compare_snapshots(context)
        mock_compare_snapshots.assert_called_once_with("snap1", "snap2", 0.5, None)

    @patch("tflens_explorer.handlers.compare_handlers.compare_snapshots")
    def test_with_percent_kwarg(self, mock_compare_snapshots):
        session = AppSession()
        context = self.make_context(session, ["snap1", "snap2", "percent=10"])
        handle_compare_snapshots(context)
        mock_compare_snapshots.assert_called_once_with("snap1", "snap2", None, 10)