from __future__ import annotations

from unittest.mock import MagicMock

from run_agent import AIAgent


def test_shutdown_memory_provider_uses_session_messages_and_notifies_vault_compiler():
    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.context_compressor = MagicMock()
    agent._session_messages = [{"role": "user", "content": "persist this session"}]
    agent._vault_knowledge = MagicMock()
    agent.session_id = "sess-123"

    AIAgent.shutdown_memory_provider(agent)

    agent._memory_manager.on_session_end.assert_called_once_with(agent._session_messages)
    agent.context_compressor.on_session_end.assert_called_once_with("sess-123", agent._session_messages)
    agent._vault_knowledge.on_session_end.assert_called_once_with(agent._session_messages)
