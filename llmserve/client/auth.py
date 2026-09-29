"""Bearer auth for backends that need an API key (ADR-0010).

`server.api_key_env` names the environment variable holding the token; the harness never
stores or logs the token itself. Most backends need no auth, so the common case is `{}`.
"""

from __future__ import annotations

import os

from llmserve.config.schema import Server


class AuthEnvError(RuntimeError):
    """The config names an API-key variable that is not set in the environment."""


def auth_headers(server: Server) -> dict[str, str]:
    """Headers to attach to every request for this server.

    Raises `AuthEnvError` when `server.api_key_env` is set but the variable is missing or
    empty — failing loudly beats sending unauthenticated requests and debugging 401s.
    """
    name = server.api_key_env
    if name is None:
        return {}
    value = os.environ.get(name, "")
    if not value:
        raise AuthEnvError(f"server.api_key_env is {name!r} but ${name} is not set")
    return {"Authorization": f"Bearer {value}"}
