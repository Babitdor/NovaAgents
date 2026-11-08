"""Synthetic comparison of request size and retained details, without provider calls."""

import json
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from novaagents import ContextConfig, ContextMiddleware
from novaagents.middleware.summarization import SummarizationMiddleware
from tests.unit_tests.middleware.test_context_controller import count, history, request
from tests.unit_tests.middleware.test_summarization_middleware import MockBackend, make_mock_model


@pytest.mark.benchmark
@pytest.mark.parametrize("mode", ["nova", "inherited"])
def test_synthetic_context_projection(benchmark, monkeypatch, mode):
    messages = history()
    messages[0] = AIMessage(content="exact signature: connect(host, *, port=8123)\n" + "history " * 150, id="m0")
    req = request(messages)
    backend = MockBackend()
    model = make_mock_model("A brief connection summary.")
    model.profile = {"max_input_tokens": 100000}
    sent = []

    def handler(current):
        sent.append(current)
        return ModelResponse(result=[AIMessage(content="done")])

    if mode == "nova":
        middleware = ContextMiddleware(model, backend, ContextConfig(input_budget=3000, token_counter=count))
        monkeypatch.setattr(middleware.accounting, "_create_summary", lambda _selected: "A brief connection summary.")
    else:
        middleware = SummarizationMiddleware(model, backend=backend, trigger=("tokens", 3000), keep=("messages", 6), token_counter=count)
        monkeypatch.setattr(middleware, "_create_summary", lambda _selected: "A brief connection summary.")

    def run():
        sent.clear()
        backend.write_calls.clear()
        backend.edit_calls.clear()
        return middleware.wrap_model_call(req, handler)

    result = benchmark(run)
    projected_units = count(sent[-1].messages)
    assert projected_units < count(messages)
    benchmark.extra_info["original_units"] = count(messages)
    benchmark.extra_info["projected_units"] = projected_units
    if mode == "nova":
        blocks = result.command.update["context_blocks"]
        block_id = next(iter(blocks))
        runtime = SimpleNamespace(state={"messages": messages, "context_blocks": blocks}, tool_call_id="recall")
        recall = next(t for t in middleware.tools if t.name == "uncollapse")
        retained = json.loads(recall.func(block_id, runtime, full=True))
        exact = "connect(host, *, port=8123)" in json.dumps(retained)
    else:
        exact = any("connect(host, *, port=8123)" in text for _, text in backend.write_calls)
    assert exact
    benchmark.extra_info["exact_detail_retrieved"] = exact
