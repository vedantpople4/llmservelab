from __future__ import annotations

import pytest

from llmserve.client.auth import AuthEnvError, auth_headers
from llmserve.config.schema import Server

KEY = "LLMSERVE_TEST_KEY"


def server(**kwargs: object) -> Server:
    base: dict[str, object] = {"kind": "nim", "endpoint": "https://example.com/v1", "model": "m"}
    base.update(kwargs)
    return Server.model_validate(base)


def test_no_api_key_configured_means_no_headers() -> None:
    assert auth_headers(server()) == {}


def test_api_key_is_read_from_the_named_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY, "s3cret")
    assert auth_headers(server(api_key_env=KEY)) == {"Authorization": "Bearer s3cret"}


def test_missing_variable_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY, raising=False)
    with pytest.raises(AuthEnvError, match=KEY):
        auth_headers(server(api_key_env=KEY))


def test_empty_variable_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY, "")
    with pytest.raises(AuthEnvError, match=KEY):
        auth_headers(server(api_key_env=KEY))
