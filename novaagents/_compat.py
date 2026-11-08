"""Compatibility shims for dependency version skew."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.agents import middleware

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentMiddleware


def get_model_refusal_middleware() -> type[AgentMiddleware]:
    """Return required refusal handling or fail with an actionable dependency error.

    Raises:
        ImportError: The installed LangChain cannot provide refusal handling.
    """
    refusal = getattr(middleware, "ModelRefusalMiddleware", None)
    if refusal is None:
        msg = "NovaAgents requires LangChain with ModelRefusalMiddleware. Run `uv sync --locked` or reinstall NovaAgents and its dependencies."
        raise ImportError(msg)
    return refusal
