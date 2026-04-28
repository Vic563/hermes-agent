from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MEMORY_BLOCK_RE = re.compile(
    r"<(?:supermemory-context|memory-context)>.*?</(?:supermemory-context|memory-context)>",
    re.DOTALL | re.IGNORECASE,
)
_MULTI_WS_RE = re.compile(r"\n{3,}")


class VaultKnowledgeCompiler:
    """Write Hermes session captures into an Obsidian vault.

    This is a lightweight, file-first compiler focused on:
    - raw session capture under raw/hermes-sessions/
    - append-only daily summaries under wiki/daily/
    - compiler activity logging under wiki/_compiler-log.md
    - master-index maintenance for daily notes

    It intentionally does not occupy Hermes's external memory-provider slot.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        vault_path: str = "",
        session_id: str = "",
        platform: str = "cli",
        capture_precompress: bool = True,
        capture_session_end: bool = True,
        compile_to_daily: bool = True,
        update_master_index: bool = True,
        max_highlights: int = 8,
        excerpt_chars: int = 220,
    ) -> None:
        self.enabled = bool(enabled and str(vault_path).strip())
        self.vault_path = Path(vault_path).expanduser() if str(vault_path).strip() else None
        self.platform = str(platform or "cli")
        self.capture_precompress = bool(capture_precompress)
        self.capture_session_end = bool(capture_session_end)
        self.compile_to_daily = bool(compile_to_daily)
        self.update_master_index = bool(update_master_index)
        self.max_highlights = max(3, int(max_highlights or 8))
        self.excerpt_chars = max(80, int(excerpt_chars or 220))

        self._lock = threading.Lock()
        self._session_id = ""
        self._session_started_at = self._now()
        self._precompress_snapshots: List[Dict[str, Any]] = []
        self._final_summary: Optional[Dict[str, Any]] = None
        self._session_closed = False
        self._last_precompress_digest = ""

        if self.enabled and self.vault_path is not None:
            self.vault_path.mkdir(parents=True, exist_ok=True)
        self.reset_session(session_id or "")

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any] | None,
        *,
        session_id: str,
        platform: str = "cli",
    ) -> "VaultKnowledgeCompiler | None":
        section = (config or {}).get("vault_knowledge", {}) if isinstance(config, dict) else {}
        if not isinstance(section, dict):
            section = {}
        enabled = bool(section.get("enabled", False))
        vault_path = str(section.get("vault_path", "") or "").strip()
        if not enabled or not vault_path:
            return None
        return cls(
            enabled=enabled,
            vault_path=vault_path,
            session_id=session_id,
            platform=platform,
            capture_precompress=section.get("capture_precompress", True),
            capture_session_end=section.get("capture_session_end", True),
            compile_to_daily=section.get("compile_to_daily", True),
            update_master_index=section.get("update_master_index", True),
            max_highlights=section.get("max_highlights", 8),
            excerpt_chars=section.get("excerpt_chars", 220),
        )

    def reset_session(self, session_id: str) -> None:
        self._session_id = str(session_id or "").strip()
        self._session_started_at = self._now()
        self._precompress_snapshots = []
        self._final_summary = None
        self._session_closed = False
        self._last_precompress_digest = ""

    def on_pre_compress(self, messages: List[Dict[str, Any]], *, focus_topic: str | None = None) -> None:
        if not self.enabled or not self.capture_precompress:
            return
        cleaned = self._clean_messages(messages)
        if not cleaned:
            return
        digest = self._messages_digest(cleaned)
        with self._lock:
            if digest and digest == self._last_precompress_digest:
                return
            self._last_precompress_digest = digest
            summary = self._build_summary(cleaned)
            summary["focus_topic"] = (focus_topic or "").strip()
            summary["captured_at"] = self._timestamp()
            summary["type"] = "pre-compress"
            self._precompress_snapshots.append(summary)
            self._write_session_note()
            if self.compile_to_daily:
                self._append_daily_event(summary, final=False)
            self._append_compiler_log(summary, final=False)
            if self.update_master_index:
                self._ensure_daily_in_index()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self.enabled or not self.capture_session_end:
            return
        with self._lock:
            if self._session_closed:
                return
            cleaned = self._clean_messages(messages)
            if not cleaned and not self._precompress_snapshots:
                return
            self._final_summary = self._build_summary(cleaned)
            self._final_summary["captured_at"] = self._timestamp()
            self._final_summary["type"] = "session-end"
            self._write_session_note(final_messages=cleaned)
            if self.compile_to_daily:
                self._append_daily_event(self._final_summary, final=True)
            self._append_compiler_log(self._final_summary, final=True)
            if self.update_master_index:
                self._ensure_daily_in_index()
            self._session_closed = True

    def _now(self) -> datetime:
        return datetime.now().astimezone()

    def _today(self) -> str:
        return self._now().strftime("%Y-%m-%d")

    def _timestamp(self) -> str:
        return self._now().strftime("%Y-%m-%d %H:%M %Z")

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                    elif "content" in item:
                        parts.append(self._extract_text(item.get("content")))
            return "\n".join(p for p in parts if p)
        if isinstance(content, dict):
            if "text" in content:
                return self._extract_text(content.get("text"))
            if "content" in content:
                return self._extract_text(content.get("content"))
            return json.dumps(content, ensure_ascii=False)
        return str(content or "")

    def _clean_text(self, text: Any) -> str:
        raw = self._extract_text(text)
        raw = _MEMORY_BLOCK_RE.sub("", raw)
        raw = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
        raw = _MULTI_WS_RE.sub("\n\n", raw)
        return raw.strip()

    def _clean_messages(self, messages: List[Dict[str, Any]] | None) -> List[Dict[str, str]]:
        cleaned: List[Dict[str, str]] = []
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip()
            if role not in {"user", "assistant", "tool"}:
                continue
            if role == "tool":
                tool_name = str(msg.get("tool_name") or "").strip() or "tool"
                cleaned.append({"role": "tool", "content": tool_name})
                continue
            content = self._clean_text(msg.get("content", ""))
            if content:
                cleaned.append({"role": role, "content": content})
        return cleaned

    def _messages_digest(self, messages: List[Dict[str, str]]) -> str:
        try:
            canonical = json.dumps(messages[-12:], ensure_ascii=False, sort_keys=True)
        except Exception:
            canonical = str(messages[-12:])
        return canonical

    def _excerpt(self, text: str) -> str:
        text = " ".join((text or "").split())
        if len(text) <= self.excerpt_chars:
            return text
        return text[: self.excerpt_chars - 3].rstrip() + "..."

    def _collect_tool_names(self, messages: List[Dict[str, str]]) -> List[str]:
        names: List[str] = []
        seen = set()
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            name = str(msg.get("content") or "").strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        return names

    def _build_summary(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        assistant_msgs = [m["content"] for m in messages if m.get("role") == "assistant"]
        highlights: List[str] = []
        for text in user_msgs + assistant_msgs:
            snippet = self._excerpt(text)
            if snippet and snippet not in highlights:
                highlights.append(snippet)
            if len(highlights) >= self.max_highlights:
                break
        return {
            "session_id": self._session_id,
            "platform": self.platform,
            "user_turns": len(user_msgs),
            "assistant_turns": len(assistant_msgs),
            "tool_names": self._collect_tool_names(messages),
            "highlights": highlights,
            "user_excerpts": [self._excerpt(t) for t in user_msgs[:4]],
            "assistant_excerpts": [self._excerpt(t) for t in assistant_msgs[:4]],
        }

    def _session_rel_path(self) -> Path:
        stamp = self._session_started_at
        return Path("raw") / "hermes-sessions" / stamp.strftime("%Y") / stamp.strftime("%m") / stamp.strftime("%d") / f"session-{self._session_id}.md"

    def _session_path(self) -> Path:
        assert self.vault_path is not None
        return self.vault_path / self._session_rel_path()

    def _daily_path(self) -> Path:
        assert self.vault_path is not None
        return self.vault_path / "wiki" / "daily" / f"{self._today()}.md"

    def _compiler_log_path(self) -> Path:
        assert self.vault_path is not None
        return self.vault_path / "wiki" / "_compiler-log.md"

    def _master_index_path(self) -> Path:
        assert self.vault_path is not None
        return self.vault_path / "wiki" / "_master-index.md"

    def _ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_file(self, path: Path, content: str) -> None:
        self._ensure_parent(path)
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    def _render_session_note(self, final_messages: List[Dict[str, str]] | None = None) -> str:
        summary = self._final_summary or self._build_summary(final_messages or [])
        status = "completed" if self._final_summary else "active"
        created = self._session_started_at.strftime("%Y-%m-%d")
        updated = self._today()
        session_rel = self._session_rel_path().as_posix()
        daily_link = f"wiki/daily/{self._today()}"
        lines = [
            "---",
            f"title: Hermes Session {self._session_id}",
            f"created: {created}",
            f"updated: {updated}",
            "type: raw-session",
            f"tags: [raw, hermes-session, {self.platform}]",
            f"session_id: {self._session_id}",
            f"platform: {self.platform}",
            f"status: {status}",
            "---",
            "",
            f"# Hermes Session {self._session_id}",
            "",
            "## Session Metadata",
            f"- Platform: {self.platform}",
            f"- Session ID: {self._session_id}",
            f"- Started: {self._session_started_at.strftime('%Y-%m-%d %H:%M %Z')}",
            f"- Updated: {self._timestamp()}",
            f"- Daily Note: [[{daily_link}]]",
            "",
        ]
        if summary.get("tool_names"):
            lines.extend([
                "## Tool Usage",
                f"- {', '.join(summary['tool_names'])}",
                "",
            ])
        for idx, snapshot in enumerate(self._precompress_snapshots, start=1):
            lines.extend([
                f"## Pre-Compression Capture {idx}",
                f"- Captured: {snapshot.get('captured_at', self._timestamp())}",
            ])
            if snapshot.get("focus_topic"):
                lines.append(f"- Focus: {snapshot['focus_topic']}")
            if snapshot.get("tool_names"):
                lines.append(f"- Tools: {', '.join(snapshot['tool_names'])}")
            lines.extend(["", "### Highlights"])
            for item in snapshot.get("highlights", []):
                lines.append(f"- {item}")
            lines.append("")
        if self._final_summary:
            lines.extend([
                "## Final Session Summary",
                f"- Finalized: {self._final_summary.get('captured_at', self._timestamp())}",
                f"- User turns: {self._final_summary.get('user_turns', 0)}",
                f"- Assistant turns: {self._final_summary.get('assistant_turns', 0)}",
            ])
            if self._final_summary.get("tool_names"):
                lines.append(f"- Tools: {', '.join(self._final_summary['tool_names'])}")
            lines.extend(["", "### Highlights"])
            for item in self._final_summary.get("highlights", []):
                lines.append(f"- {item}")
            lines.extend(["", "### Transcript Excerpts", "#### User"])
            for item in self._final_summary.get("user_excerpts", []):
                lines.append(f"- {item}")
            lines.extend(["", "#### Assistant"])
            for item in self._final_summary.get("assistant_excerpts", []):
                lines.append(f"- {item}")
            lines.extend(["", "## Sources", f"- [[{session_rel[:-3]}]]", f"- [[{daily_link}]]"])
        return "\n".join(lines).rstrip() + "\n"

    def _write_session_note(self, final_messages: List[Dict[str, str]] | None = None) -> None:
        if not self._session_id:
            return
        path = self._session_path()
        self._ensure_parent(path)
        path.write_text(self._render_session_note(final_messages), encoding="utf-8")

    def _ensure_daily_in_index(self) -> None:
        path = self._master_index_path()
        self._ensure_file(path, "# Knowledge Base Index\n\n## Daily\n")
        link_line = f"- [[daily/{self._today()}]] - Automatic Hermes session capture summaries."
        content = path.read_text(encoding="utf-8")
        if link_line in content:
            return
        if "## Daily" in content:
            updated = content.replace("## Daily\n", f"## Daily\n{link_line}\n", 1)
        else:
            updated = content.rstrip() + f"\n\n## Daily\n{link_line}\n"
        path.write_text(updated, encoding="utf-8")

    def _append_daily_event(self, summary: Dict[str, Any], *, final: bool) -> None:
        path = self._daily_path()
        self._ensure_file(path, f"# Daily Log: {self._today()}\n")
        marker = f"<!-- vault-knowledge:{self._session_id}:{'final' if final else f'pre{len(self._precompress_snapshots)}'} -->"
        content = path.read_text(encoding="utf-8")
        if marker in content:
            return
        session_rel = self._session_rel_path().as_posix()[:-3]
        if final:
            heading = f"## Hermes Session {self._session_id} finalized"
            lines = [
                heading,
                "",
                f"- Captured: {summary.get('captured_at', self._timestamp())}",
                f"- Platform: {self.platform}",
                f"- User turns: {summary.get('user_turns', 0)}",
                f"- Assistant turns: {summary.get('assistant_turns', 0)}",
                f"- Source: [[{session_rel}]]",
            ]
            if summary.get("tool_names"):
                lines.append(f"- Tools: {', '.join(summary['tool_names'])}")
            lines.extend(["", "### Highlights"])
            for item in summary.get("highlights", []):
                lines.append(f"- {item}")
        else:
            heading = f"## Pre-Compression Capture for {self._session_id}"
            lines = [
                heading,
                "",
                f"- Captured: {summary.get('captured_at', self._timestamp())}",
                f"- Platform: {self.platform}",
                f"- Source: [[{session_rel}]]",
            ]
            if summary.get("focus_topic"):
                lines.append(f"- Focus: {summary['focus_topic']}")
            if summary.get("tool_names"):
                lines.append(f"- Tools: {', '.join(summary['tool_names'])}")
            lines.extend(["", "### Highlights"])
            for item in summary.get("highlights", []):
                lines.append(f"- {item}")
        block = "\n" + "\n".join(lines) + f"\n\n{marker}\n"
        path.write_text(content.rstrip() + block, encoding="utf-8")

    def _append_compiler_log(self, summary: Dict[str, Any], *, final: bool) -> None:
        path = self._compiler_log_path()
        self._ensure_file(path, "# SSNC Wiki Compiler Log\n")
        marker = f"<!-- vault-knowledge-log:{self._session_id}:{'final' if final else f'pre{len(self._precompress_snapshots)}'} -->"
        content = path.read_text(encoding="utf-8")
        if marker in content:
            return
        session_rel = self._session_rel_path().as_posix()
        action = "session_end" if final else "pre_compress"
        lines = [
            f"## [{self._timestamp()}] {action} | {self._session_id}",
            f"- Wrote `{session_rel}`",
            f"- Updated `wiki/daily/{self._today()}.md`",
            "- Updated `wiki/_master-index.md`",
            marker,
        ]
        path.write_text(content.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
