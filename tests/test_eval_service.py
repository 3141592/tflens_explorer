import torch

from tflens_explorer.services.eval_service import rank_expected_tokens


class FakeModel:
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


class FakeModelExpectedSecond(FakeModel):
    def __call__(self, prompt, prepend_bos=True):
        logits = torch.zeros(1, 1, 200)

        logits[0, -1, 70] = 10.0  
        logits[0, -1, 20] = 9.0   
        logits[0, -1, 40] = 8.0   # " couch" highest
        logits[0, -1, 10] = 7.0   # " floor"
        logits[0, -1, 30] = 5.0
        logits[0, -1, 50] = 4.0
        logits[0, -1, 60] = 1.0

        return logits


class FakeModelExpectedLowRank(FakeModel):
    def __call__(self, prompt, prepend_bos=True):
        logits = torch.zeros(1, 1, 200)

        logits[0, -1, 60] = 10.0  # " bench" highest
        logits[0, -1, 90] = 9.0
        logits[0, -1, 80] = 7.0
        logits[0, -1, 70] = 5.0
        logits[0, -1, 50] = 4.0

        logits[0, -1, 10] = 1.0   # " floor"
        return logits


def test_rank_expected_tokens_finds_best_expected_rank():
    model = FakeModel()

    eval_case = {
        "name": "dog_floor",
        "prompt": "The dog sat on the",
        "prepend_bos": True,
        "expected_next_tokens": [" floor", " couch", " bed"],
    }

    result = rank_expected_tokens(model, eval_case)

    assert result["next_token"] == " floor"
    assert result["next_token_id"] == 10
    assert result["expected_token_best_rank"] == 1
    assert result["expected_in_top_1"] is True
    assert result["expected_in_top_5"] is True


def test_rank_expected_token_is_not_top_1_but_is_top_5():
    model = FakeModelExpectedSecond()

    eval_case = {
        "name": "dog_floor",
        "prompt": "The dog sat on the",
        "prepend_bos": True,
        "expected_next_tokens": [" floor", " couch", " bed"],
    }

    result = rank_expected_tokens(model, eval_case)

    assert result["next_token"] == " side"
    assert result["next_token_id"] == 70
    assert result["expected_token_best_rank"] == 3
    assert result["expected_in_top_1"] is False
    assert result["expected_in_top_5"] is True


def test_rank_expected_token_is_not_in_top_5():
    model = FakeModelExpectedLowRank()

    eval_case = {
        "name": "dog_floor",
        "prompt": "The dog sat on the",
        "prepend_bos": True,
        "expected_next_tokens": [" floor", " couch", " bed"],
    }

    result = rank_expected_tokens(model, eval_case)

    assert result["next_token"] == " edge"
    assert result["next_token_id"] == 60
    assert result["expected_token_best_rank"] > 5
    assert result["expected_in_top_1"] is False
    assert result["expected_in_top_5"] is False