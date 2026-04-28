from __future__ import annotations

from datetime import datetime

from agent.vault_knowledge import VaultKnowledgeCompiler


class TestVaultKnowledgeCompiler:
    def test_session_end_and_precompress_write_into_vault(self, tmp_path):
        vault = tmp_path / "vault"
        (vault / "wiki").mkdir(parents=True)
        (vault / "raw").mkdir(parents=True)
        (vault / "wiki" / "_master-index.md").write_text(
            "# Knowledge Base Index\n\n## Daily\n",
            encoding="utf-8",
        )

        compiler = VaultKnowledgeCompiler(
            enabled=True,
            vault_path=str(vault),
            session_id="sess-123",
            platform="cli",
        )

        messages = [
            {"role": "user", "content": "Please design the session-to-vault compiler."},
            {"role": "assistant", "content": "I will create the vault integration hooks."},
            {"role": "tool", "tool_name": "read_file", "content": '{"ok": true}'},
            {"role": "user", "content": "Make sure it writes to Obsidian automatically."},
            {"role": "assistant", "content": "Understood. I will wire session end and pre-compress capture."},
        ]

        compiler.on_pre_compress(messages, focus_topic="vault capture")
        compiler.on_session_end(messages)

        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        session_note = vault / "raw" / "hermes-sessions" / datetime.now().astimezone().strftime("%Y") / datetime.now().astimezone().strftime("%m") / datetime.now().astimezone().strftime("%d") / "session-sess-123.md"
        daily_note = vault / "wiki" / "daily" / f"{today}.md"
        compiler_log = vault / "wiki" / "_compiler-log.md"
        index_path = vault / "wiki" / "_master-index.md"

        assert session_note.exists()
        session_text = session_note.read_text(encoding="utf-8")
        assert "Pre-Compression Capture 1" in session_text
        assert "Final Session Summary" in session_text
        assert "read_file" in session_text

        assert daily_note.exists()
        daily_text = daily_note.read_text(encoding="utf-8")
        assert "sess-123" in daily_text
        assert "session-sess-123" in daily_text

        assert compiler_log.exists()
        log_text = compiler_log.read_text(encoding="utf-8")
        assert "sess-123" in log_text

        index_text = index_path.read_text(encoding="utf-8")
        assert f"[[daily/{today}]]" in index_text
