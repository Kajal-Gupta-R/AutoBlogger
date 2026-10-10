import httpx
import pytest
from groq import AuthenticationError, RateLimitError
from tenacity import wait_none

from backend import nodes


def make_error(cls, status):
    request = httpx.Request("POST", "https://api.groq.com")
    response = httpx.Response(status, request=request)
    return cls("error", response=response, body=None)


class FlakyLLM:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return type("Reply", (), {"content": "ok"})()


def test_retries_rate_limit_then_succeeds(monkeypatch):
    llm = FlakyLLM([make_error(RateLimitError, 429)] * 2)
    monkeypatch.setattr(nodes, "llm", llm)
    assert nodes.ask.retry_with(wait=wait_none())("hi") == "ok"
    assert llm.calls == 3


def test_does_not_retry_bad_key(monkeypatch):
    llm = FlakyLLM([make_error(AuthenticationError, 401)])
    monkeypatch.setattr(nodes, "llm", llm)
    with pytest.raises(AuthenticationError):
        nodes.ask.retry_with(wait=wait_none())("hi")
    assert llm.calls == 1