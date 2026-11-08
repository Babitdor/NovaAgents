"""Exercise the public NovaAgents entry point without provider credentials."""

from langchain_core.messages import AIMessage

from novaagents import ContextConfig, create_nova_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


def test_nova_agent_invocation() -> None:
    agent = create_nova_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="Hello from Nova")])),
        context_management=ContextConfig(input_budget=100000),
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

    assert result["messages"][-1].content == "Hello from Nova"
    assert agent.config["metadata"]["ls_integration"] == "novaagents"
