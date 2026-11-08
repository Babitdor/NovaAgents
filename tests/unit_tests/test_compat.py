"""Required safety middleware must never disappear silently."""

import pytest
from langchain.agents import middleware

from novaagents._compat import get_model_refusal_middleware


def test_refusal_middleware_matches_installed_dependency():
    assert get_model_refusal_middleware() is middleware.ModelRefusalMiddleware


def test_missing_refusal_middleware_reports_dependency_repair(monkeypatch):
    monkeypatch.delattr(middleware, "ModelRefusalMiddleware")
    with pytest.raises(ImportError, match="reinstall NovaAgents"):
        get_model_refusal_middleware()
