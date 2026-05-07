from types import SimpleNamespace

class FakeTokenizer:
    def encode(self, text):
        return [len(text)]  # simple, deterministic

    def decode(self, token_ids):
        return "Hello"

class FakeModel:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.cfg = SimpleNamespace(d_vocab=50257)

    def run_with_cache(self, tokens, prepend_bos=True):
        logits = [[0.1, 0.2]]
        cache = {"fake_key": "fake_value"}
        return logits, cache

    def to_tokens(self, prompt, prepend_bos=True):
        return [1, 2, 3]

