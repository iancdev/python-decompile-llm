from types import SimpleNamespace
import openai
from decompilellm.api import call_llm


class StubCompletions:
    def create(self, **kwargs):
        if kwargs.get("stream"):
            chunks = ["foo", "bar"]
            for part in chunks:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=part))])
        else:
            content = "foobar"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")])


class StubChat:
    def __init__(self):
        self.completions = StubCompletions()


class StubClient:
    def __init__(self, *args, **kwargs):
        self.chat = StubChat()


def test_call_llm_stream(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: StubClient())
    result, err = call_llm("key", "model", "sys", "prompt", "openai", True)
    assert err is None
    assert result == "foobar"


def test_call_llm_non_stream(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: StubClient())
    result, err = call_llm("key", "model", "sys", "prompt", "openai", False)
    assert err is None
    assert result == "foobar"
