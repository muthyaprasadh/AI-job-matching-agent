from job_agent.agent import JobAgent
from job_agent.db import Store


def test_agent_executes_tool_and_persists_conversation(tmp_path, monkeypatch):
    store = Store(tmp_path / "agent.sqlite")
    agent = JobAgent(store, api_key="test-key")
    assert store.get_profile("default") == agent.profile
    responses = iter(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_jobs",
                            "arguments": '{"query":"Python","limit":3}',
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "I found no saved jobs for that query."},
        ]
    )
    monkeypatch.setattr(agent, "_request_model", lambda messages: next(responses))

    result = agent.chat("Find Python jobs")

    assert result.tools_used == ["search_jobs"]
    assert result.answer == "I found no saved jobs for that query."
    history = store.get_conversation_messages(result.conversation_id)
    assert [message["role"] for message in history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    store.conn.close()


def test_agent_rejects_invalid_tool_arguments(tmp_path):
    store = Store(tmp_path / "agent.sqlite")
    agent = JobAgent(store, api_key="test-key")

    result = agent._invoke_tool("search_jobs", '{"query":"","limit":99}')

    assert "error" in result
    store.conn.close()
