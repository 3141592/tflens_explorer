import torch
from types import SimpleNamespace
from unittest.mock import patch
from tflens_explorer.services.eval_service import rank_expected_tokens


class FakeModel:
    def __init__(self):
        self.cfg = SimpleNamespace(model_name="FakeModel", d_vocab=50257)

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


@patch("tflens_explorer.services.eval_service.rank_expected_tokens")
def test_run_model_eval_returns_rank_results(mock_rank):
    model = FakeModel()

    mock_rank.return_value = {
        "next_token": " floor",
        "next_token_id": 10,
        "expected_token_best_rank": 1,
        "expected_in_top_1": True,
        "expected_in_top_5": True,
    }

    eval_case = {
        "name": "dog_floor",
        "prompt": "The dog sat on the",
        "prepend_bos": True,
        "expected_next_tokens": [" floor", " couch", " bed"],
    }
