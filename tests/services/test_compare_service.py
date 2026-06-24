"""Tests for compare_service.py."""
import torch
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from types import SimpleNamespace
from dataclasses import dataclass, field
from tflens_explorer.services.compare_service import (
    snapshots_list,
    compare_mlp,
    compare_attention,
    compare_generated,
    compare_evals,
    compare_models,
    compare_runs,
    compare_logits_details,
    compare_token_ids,
    compare_tokens,
    compare_logits_ranks,
    compare_logits_probs,
    cache_percent_diff,
    cache_diff,
    cache_mean_abs_diff,
    cache_cosine_similarity,
    is_cosine_eligible,
    head_axis_for_hook,
    compare_logits,
    compare_cache,
    compare_snapshots,
)


# ---------------------------------------------------------------------------
# Pure-computation helpers (no mocking needed)
# ---------------------------------------------------------------------------

class TestCompareLogitsDetails:
    def test_matching_top_1_top_5_overlap(self):
        # compare_logits_details pops the first element (rank 1) before comparing
        # so with 5 entries, it compares indices 1-4 (the remaining 4 entries)
        logits_a = [
            {"rank": 1, "token": " hello", "prob": 0.5, "value": 1.0},
            {"rank": 2, "token": " world", "prob": 0.3, "value": 0.8},
            {"rank": 3, "token": " foo", "prob": 0.1, "value": 0.5},
            {"rank": 4, "token": " bar", "prob": 0.05, "value": 0.3},
            {"rank": 5, "token": " baz", "prob": 0.03, "value": 0.2},
        ]
        logits_b = [
            {"rank": 1, "token": " hello", "prob": 0.4, "value": 0.9},
            {"rank": 2, "token": " world", "prob": 0.25, "value": 0.7},
            {"rank": 3, "token": " foo", "prob": 0.15, "value": 0.6},
            {"rank": 4, "token": " bar", "prob": 0.08, "value": 0.4},
            {"rank": 5, "token": " baz", "prob": 0.03, "value": 0.2},
        ]
        top_1, overlap = compare_logits_details(logits_a, logits_b)
        # After pop(0), rank 1 entries are removed; top_1 uses the new first elements
        assert top_1 == [" world", " world"]
        # 4 remaining items in each list all match token-wise → overlap = 4
        assert overlap == 4

    def test_no_top_5_overlap(self):
        logits_a = [
            {"rank": 1, "token": " hello", "prob": 0.5, "value": 1.0},
            {"rank": 2, "token": " alpha", "prob": 0.2, "value": 0.8},
            {"rank": 3, "token": " beta", "prob": 0.1, "value": 0.5},
            {"rank": 4, "token": " gamma", "prob": 0.05, "value": 0.3},
            {"rank": 5, "token": " delta", "prob": 0.03, "value": 0.2},
        ]
        logits_b = [
            {"rank": 1, "token": " world", "prob": 0.4, "value": 0.9},
            {"rank": 2, "token": " zeta", "prob": 0.3, "value": 0.8},
            {"rank": 3, "token": " eta", "prob": 0.2, "value": 0.7},
            {"rank": 4, "token": " theta", "prob": 0.1, "value": 0.5},
            {"rank": 5, "token": " iota", "prob": 0.05, "value": 0.3},
        ]
        top_1, overlap = compare_logits_details(logits_a, logits_b)
        # After pop(0), top_1 uses the new first elements
        assert top_1 == [" alpha", " zeta"]
        assert overlap == 0


class TestCompareTokenIds:
    def test_equal_length_all_match(self):
        # compare_token_ids pops the first element before comparing
        tokens1 = [SimpleNamespace(token_id=1), SimpleNamespace(token_id=2)]
        tokens2 = [SimpleNamespace(token_id=1), SimpleNamespace(token_id=2)]
        result = compare_token_ids(tokens1, tokens2)
        # After pop(0), only the second element remains to compare
        assert result == [True]

    def test_equal_length_some_mismatch(self):
        tokens1 = [SimpleNamespace(token_id=1), SimpleNamespace(token_id=2)]
        tokens2 = [SimpleNamespace(token_id=1), SimpleNamespace(token_id=99)]
        result = compare_token_ids(tokens1, tokens2)
        # After pop(0), only index 1 is compared → 2 != 99 → False
        assert result == [False]

    def test_different_lengths(self):
        tokens1 = [SimpleNamespace(token_id=1), SimpleNamespace(token_id=2)]
        tokens2 = [SimpleNamespace(token_id=1)]
        result = compare_token_ids(tokens1, tokens2)
        # After pop(0) from both: tokens1 has [2], tokens2 is empty
        # len(tokens2) <= index (0) → False
        assert result == [False]


class TestCompareTokens:
    def test_equal_length_all_match(self):
        tokens1 = [SimpleNamespace(token=" hello"), SimpleNamespace(token=" world")]
        tokens2 = [SimpleNamespace(token=" hello"), SimpleNamespace(token=" world")]
        result = compare_tokens(tokens1, tokens2)
        assert result == [True, True]

    def test_equal_length_some_mismatch(self):
        tokens1 = [SimpleNamespace(token=" hello"), SimpleNamespace(token=" world")]
        tokens2 = [SimpleNamespace(token=" hello"), SimpleNamespace(token=" foo")]
        result = compare_tokens(tokens1, tokens2)
        assert result == [True, False]

    def test_different_lengths(self):
        tokens1 = [SimpleNamespace(token=" hello"), SimpleNamespace(token=" world")]
        tokens2 = [SimpleNamespace(token=" hello")]
        result = compare_tokens(tokens1, tokens2)
        assert result == [True, False]


class TestCompareLogitsRanks:
    def test_matching_tokens_shows_rankings(self, capsys):
        logits1 = [
            {"index": 1, "prob": 0.5, "token": " hello"},
            {"index": 2, "prob": 0.3, "token": " world"},
        ]
        logits2 = [
            {"index": 2, "prob": 0.4, "token": " hello"},
            {"index": 1, "prob": 0.3, "token": " world"},
        ]
        compare_logits_ranks(logits1, logits2)
        captured = capsys.readouterr()
        assert "hello" in captured.out
        assert "world" in captured.out
        assert "ΔRank" in captured.out

    def test_no_common_tokens(self, capsys):
        logits1 = [{"index": 1, "prob": 0.5, "token": " hello"}]
        logits2 = [{"index": 1, "prob": 0.5, "token": " world"}]
        compare_logits_ranks(logits1, logits2)
        captured = capsys.readouterr()
        assert "No tokens in common." in captured.out


class TestCompareLogitsProbs:
    def test_matching_tokens_shows_probs(self, capsys):
        logits1 = [
            {"index": 1, "prob": 0.5, "token": " hello"},
            {"index": 2, "prob": 0.3, "token": " world"},
        ]
        logits2 = [
            {"index": 2, "prob": 0.4, "token": " hello"},
            {"index": 1, "prob": 0.3, "token": " world"},
        ]
        compare_logits_probs(logits1, logits2)
        captured = capsys.readouterr()
        assert "hello" in captured.out
        assert "world" in captured.out
        assert "ΔProb" in captured.out

    def test_no_common_tokens(self, capsys):
        logits1 = [{"index": 1, "prob": 0.5, "token": " hello"}]
        logits2 = [{"index": 1, "prob": 0.5, "token": " world"}]
        compare_logits_probs(logits1, logits2)
        captured = capsys.readouterr()
        assert "No tokens in common." in captured.out


class TestCachePercentDiff:
    def test_zero_limit_returns_true(self):
        assert cache_percent_diff(1.0, 2.0, 0) is True

    def test_below_limit_returns_false(self):
        assert cache_percent_diff(100.0, 105.0, 10) is False

    def test_above_limit_returns_true(self):
        assert cache_percent_diff(100.0, 120.0, 10) is True

    def test_equal_values_returns_false(self):
        assert cache_percent_diff(5.0, 5.0, 5) is False

    def test_mean1_is_zero(self):
        assert cache_percent_diff(0.0, 5.0, 1) is True

    def test_mean2_is_zero(self):
        assert cache_percent_diff(5.0, 0.0, 1) is True


class TestCacheDiff:
    def test_zero_limit_returns_true(self):
        assert cache_diff(0.5, 0) is True

    def test_below_limit_returns_false(self):
        assert cache_diff(0.1, 1.0) is False

    def test_above_limit_returns_true(self):
        assert cache_diff(2.0, 1.0) is True


class TestCacheMeanAbsDiff:
    def test_int_tensors(self):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([2.0, 3.0, 4.0])
        result = cache_mean_abs_diff(t1, t2)
        assert result == 1.0

    def test_int_tensors_are_converted(self):
        t1 = torch.tensor([1, 2, 3])
        t2 = torch.tensor([2, 3, 4])
        result = cache_mean_abs_diff(t1, t2)
        assert result == 1.0

    def test_identical_tensors(self):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 3.0])
        result = cache_mean_abs_diff(t1, t2)
        assert result == 0.0


class TestCacheCosineSimilarity:
    def test_identical_tensors(self):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 3.0])
        result = cache_cosine_similarity(t1, t2)
        assert result is not None
        assert abs(result - 1.0) < 1e-6

    def test_opposite_tensors(self):
        t1 = torch.tensor([1.0, 0.0])
        t2 = torch.tensor([-1.0, 0.0])
        result = cache_cosine_similarity(t1, t2)
        assert result is not None
        assert abs(result - (-1.0)) < 1e-6

    def test_filters_sentinel_values(self):
        t1 = torch.tensor([1.0, -1e30, 2.0])
        t2 = torch.tensor([2.0, -1e30, 4.0])
        result = cache_cosine_similarity(t1, t2)
        assert result is not None
        # Should only compare [1.0, 2.0] vs [2.0, 4.0]

    def test_empty_after_masking(self):
        t1 = torch.tensor([-1e30])
        t2 = torch.tensor([-1e30])
        result = cache_cosine_similarity(t1, t2)
        assert result is None

    def test_int_tensors_are_converted(self):
        t1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        t2 = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = cache_cosine_similarity(t1, t2)
        assert abs(result - 1.0) < 1e-6


class TestIsCosineEligible:
    def test_eligible(self):
        t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert is_cosine_eligible(t1, t2) is True

    def test_shape_mismatch(self):
        t1 = torch.tensor([[1.0, 2.0]])
        t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert is_cosine_eligible(t1, t2) is False

    def test_empty_tensor(self):
        t1 = torch.tensor([])
        t2 = torch.tensor([])
        assert is_cosine_eligible(t1, t2) is False

    def test_non_float(self):
        t1 = torch.tensor([1, 2], dtype=torch.int32)
        t2 = torch.tensor([1, 2], dtype=torch.int32)
        assert is_cosine_eligible(t1, t2) is False


class TestHeadAxisForHook:
    def test_head_on_axis_1(self):
        tensor = torch.zeros(5, 12, 64)  # [seq, head, d_head]
        assert head_axis_for_hook("blocks.0.attn.hook_result", tensor, 12) == 1

    def test_head_on_axis_0(self):
        tensor = torch.zeros(12, 5, 64)  # [head, query, key]
        assert head_axis_for_hook("some_hook", tensor, 12) == 0

    def test_no_head_dim(self):
        tensor = torch.zeros(5, 768)
        assert head_axis_for_hook("blocks.0.hook_resid_pre", tensor, 12) is None


# ---------------------------------------------------------------------------
# Stub commands (simple delegation / print)
# ---------------------------------------------------------------------------

class TestStubFunctions:
    def test_snapshots_list(self, capsys, tmp_path):
        from tflens_explorer.services.compare_service import SNAPSHOT_PATH as orig_snapshot_path

        # Create a real directory in tmp_path to avoid mocking Path entirely
        snapshot_dir = tmp_path / "snapshots"
        snapshot_subdir = snapshot_dir / "my_snapshot"
        snapshot_subdir.mkdir(parents=True, exist_ok=True)

        with patch("tflens_explorer.services.compare_service.SNAPSHOT_PATH", new=snapshot_dir):
            snapshots_list()
            captured = capsys.readouterr()
            assert "my_snapshot" in captured.out

    @pytest.mark.parametrize("func,expected", [
        (compare_mlp, "compare-mlp"),
        (compare_attention, "compare-attention"),
        (compare_generated, "compare-generated"),
        (compare_evals, "compare-evals"),
        (compare_models, "compare-models"),
        (compare_runs, "compare-runs"),
    ])
    def test_stub_prints_name(self, func, expected, capsys):
        func()
        captured = capsys.readouterr()
        assert expected in captured.out


# ---------------------------------------------------------------------------
# compare_logits – service-level
# ---------------------------------------------------------------------------

class FakeCacheSummary:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestCompareLogits:
    @patch("tflens_explorer.services.compare_service.verify_snapshot")
    @patch("tflens_explorer.services.compare_service.Snapshot")
    def test_snapshot_not_found(self, mock_snapshot_cls, mock_verify, capsys):
        mock_verify.return_value = False
        result = compare_logits("nonexistent1", "nonexistent2")
        captured = capsys.readouterr()
        assert "does not exist" in captured.out
        mock_snapshot_cls.assert_not_called()

    @patch("tflens_explorer.services.compare_service.verify_snapshot")
    @patch("tflens_explorer.services.compare_service.Snapshot")
    @patch("tflens_explorer.services.compare_service.compare_logits_details")
    @patch("tflens_explorer.services.compare_service.compare_logits_ranks")
    @patch("tflens_explorer.services.compare_service.compare_logits_probs")
    def test_valid_snapshots(
        self,
        mock_probs,
        mock_ranks,
        mock_details,
        mock_snapshot_cls,
        mock_verify,
    ):
        mock_verify.return_value = True
        mock_details.return_value = ([" hello", " hello"], 5)

        snap1 = MagicMock()
        snap1.logits = [10]
        snap1.model.name = "ModelA"
        snap1.prompt = "hello"
        snap2 = MagicMock()
        snap2.logits = [10]
        snap2.model.name = "ModelB"
        snap2.prompt = "hello"

        def load_side_effect(name):
            if name == "snap1":
                return snap1
            return snap2

        mock_snapshot_cls.load = MagicMock(side_effect=load_side_effect)
        mock_snapshot_cls.return_value = MagicMock()

        compare_logits("snap1", "snap2")
        mock_verify.assert_any_call("snap1")
        mock_verify.assert_any_call("snap2")


# ---------------------------------------------------------------------------
# compare_cache – service-level
# ---------------------------------------------------------------------------

class TestCompareCache:
    @patch("tflens_explorer.services.compare_service.verify_snapshot")
    @patch("tflens_explorer.services.compare_service.Snapshot")
    def test_snapshot_not_found(self, mock_snapshot_cls, mock_verify, capsys):
        mock_verify.return_value = False
        compare_cache("nonexistent1", "nonexistent2")
        captured = capsys.readouterr()
        assert "does not exist" in captured.out
        mock_snapshot_cls.assert_not_called()

    @patch("tflens_explorer.services.compare_service.verify_snapshot")
    @patch("tflens_explorer.services.compare_service.Snapshot")
    @patch("tflens_explorer.services.compare_service.cache_activation_summary")
    def test_valid_snapshots(self, mock_cas, mock_snapshot_cls, mock_verify):
        mock_verify.return_value = True
        snap1 = MagicMock()
        snap1.model.name = "ModelA"
        snap1.prompt = "hello"
        snap1.cache = []
        snap2 = MagicMock()
        snap2.model.name = "ModelB"
        snap2.prompt = "world"
        snap2.cache = []

        # Make Snapshot(name=...) return the pre-built mock with load as a method
        snap1.load = MagicMock()
        snap2.load = MagicMock()

        def construct_side_effect(name=None):
            if name == "snap1":
                return snap1
            return snap2

        mock_snapshot_cls.side_effect = construct_side_effect

        compare_cache("snap1", "snap2")
        snap1.load.assert_called_once()
        snap2.load.assert_called_once()
        mock_cas.assert_called_once_with(snap1.cache, snap2.cache)


# ---------------------------------------------------------------------------
# compare_snapshots – service-level
# ---------------------------------------------------------------------------

class TestCompareSnapshots:
    @patch("tflens_explorer.services.compare_service.verify_snapshot")
    @patch("tflens_explorer.services.compare_service.Snapshot")
    def test_snapshot_not_found(self, mock_snapshot_cls, mock_verify, capsys):
        mock_verify.return_value = False
        compare_snapshots("nonexistent1", "nonexistent2", None, None)
        captured = capsys.readouterr()
        assert "does not exist" in captured.out
        mock_snapshot_cls.assert_not_called()

    @patch("tflens_explorer.services.compare_service.verify_snapshot")
    @patch("tflens_explorer.services.compare_service.Snapshot.load")
    @patch("tflens_explorer.services.compare_service.compare_token_ids")
    @patch("tflens_explorer.services.compare_service.compare_tokens")
    @patch("tflens_explorer.services.compare_service.compare_logits_details")
    @patch("tflens_explorer.services.compare_service.cache_activation_summary")
    def test_valid_snapshots(
        self,
        mock_cas,
        mock_details,
        mock_comp_tokens,
        mock_comp_token_ids,
        mock_load,
        mock_verify,
        capsys,
    ):
        mock_verify.return_value = True
        mock_comp_token_ids.return_value = [True, True]
        mock_comp_tokens.return_value = [True, True]
        mock_details.return_value = ([" hello", " hello"], 5)

        snap1 = MagicMock()
        snap1.token_shape = "[1, 3]"
        snap1.logit_shape = "[1, 3, 50257]"
        snap1.tokens = []
        snap1.logits = []
        snap1.cache = [MagicMock()]
        snap1.model.name = "ModelA"
        snap1.prompt = "hello"

        snap2 = MagicMock()
        snap2.token_shape = "[1, 3]"
        snap2.logit_shape = "[1, 3, 50257]"
        snap2.tokens = []
        snap2.logits = []
        snap2.cache = [MagicMock()]
        snap2.model.name = "ModelB"
        snap2.prompt = "hello"

        def load_side_effect(name):
            if name == "snap1":
                return snap1
            return snap2

        mock_load.side_effect = load_side_effect

        compare_snapshots("snap1", "snap2", None, None)

        mock_verify.assert_any_call("snap1")
        mock_verify.assert_any_call("snap2")
        captured = capsys.readouterr()
        assert "ModelA" in captured.out
        assert "ModelB" in captured.out