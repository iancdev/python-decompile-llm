import tiktoken
from decompilellm.token_utils import get_token_count


def test_get_token_count_fallback(monkeypatch):
    def raise_error(model):
        raise Exception("no encoding")

    class DummyEncoding:
        def encode(self, text):
            return list(text)

    monkeypatch.setattr(tiktoken, "encoding_for_model", raise_error)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: DummyEncoding())
    count = get_token_count("hello", "model", "openai")
    assert count == len("hello")
