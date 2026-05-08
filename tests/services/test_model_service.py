import torch
import pytest
from types import SimpleNamespace
from unittest.mock import patch
from tflens_explorer.services.model_service import resolve_model_name
from tflens_explorer.services.model_service import load_model
from tflens_explorer.services.model_service import get_model_info


class FakeCfg:
    n_layers = 12
    n_heads = 12
    d_model = 768
    d_head = 64
    d_vocab = 50257
    n_ctx = 1024
    device = "cuda"

class FakeModel:
    def __init__(self):
        self.cfg = FakeCfg()
        device = "cuda"

    def __call__(self, prompt, prepend_bos=True):
        logits = torch.zeros(1, 1, 200)

        logits[0, -1, 10] = 9.0   # " floor" top token
        logits[0, -1, 20] = 7.0   # " ground"
        logits[0, -1, 30] = 5.0   # " bed"
        logits[0, -1, 40] = 4.0   # " couch"
        logits[0, -1, 50] = 3.0   # " back"
        logits[0, -1, 60] = 1.0   # " edge"
        logits[0, -1, 70] = 1.0
        logits[0, -1, 80] = 1.0
        logits[0, -1, 90] = 1.0
        logits[0, -1, 100] = 1.0

        return logits

    def to_str_tokens(self, token_id):
        token_id = int(token_id)
        reverse = {
            10: " floor",
            20: " ground",
            30: " bed",
            40: " couch",
            50: " back",
            60: " edge",
            70: " side",
            80: " front",
            90: " sofa",
            100: " bench",
        }
        return [reverse[token_id]]

    def to_single_token(self, token):
        if isinstance(token, list):
            token = token[0]

        mapping = {
            " floor": 10,
            " ground": 20,
            " bed": 30,
            " couch": 40,
            " back": 50,
            " edge": 60,
            " side": 70,
            " front": 80,
            " sofa": 90,
            " bench": 100,
        }
        return mapping[token]

    def generate(self, prompt, max_new_tokens=1, do_sample=False):
        return prompt

def test_resolve_model_name_uses_alias():
    assert resolve_model_name("gpt2") == "openai-community/gpt2"

@patch("tflens_explorer.services.model_service.TransformerBridge")
def test_load_model_by_alias(mock_bridge):
    fake_model = object()
    mock_bridge.boot_transformers.return_value = fake_model
    resolved_model_name = resolve_model_name("gpt2")
    model = load_model(resolved_model_name)

    mock_bridge.boot_transformers.assert_called_once_with(
        "openai-community/gpt2",
        device="cuda",
    )
    assert model is fake_model


@patch("tflens_explorer.services.model_service.TransformerBridge")
def test_load_model_by_full_model_id(mock_bridge):
    fake_model = object()
    mock_bridge.boot_transformers.return_value = fake_model
    model = load_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    mock_bridge.boot_transformers.assert_called_once_with(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device="cuda",
    )
    assert model is fake_model


def test_resolve_model_name_unknown_alias_returns_original():
    result =  resolve_model_name("nonsense_model")
    assert result == "nonsense_model" 

@patch("tflens_explorer.services.model_service.TransformerBridge")
def test_load_model_raises_exception_on_failure(mock_bridge):
    mock_bridge.boot_transformers.side_effect = RuntimeError("load failed")

    with pytest.raises(RuntimeError, match="load failed"):
        load_model("gpt2")

@patch("tflens_explorer.services.model_service.TransformerBridge")
def test_get_model_info(model):
    model = FakeModel()
    info = get_model_info(model)

    assert info["n_layers"] == 12
    assert info["n_heads"] == 12
    assert info["d_model"] == 768
    assert info["d_head"] == 64
    assert info["d_vocab"] == 50257
    assert info["n_ctx"] == 1024
    assert info["device"] == "cuda"
    

@patch("tflens_explorer.services.model_service.TransformerBridge")
def test_load_model_uses_explicit_device(mock_bridge):
    fake_model = object()
    mock_bridge.boot_transformers.return_value = fake_model

    model = load_model("openai-community/gpt2", device="cpu")

    mock_bridge.boot_transformers.assert_called_once_with(
        "openai-community/gpt2",
        device="cpu",
    )
    assert model is fake_model