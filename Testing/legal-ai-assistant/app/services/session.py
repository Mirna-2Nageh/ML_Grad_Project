"""
Session Manager — Conversation history with sliding-window compaction + disk persistence.

Stores chat history per session. When history exceeds SESSION_MAX_TURNS, compacts older
messages into a summary using the LLM. Sessions are persisted as JSON files in
SESSION_PERSIST_DIR (when SESSION_PERSIST=True) so they survive server restarts.
Idle sessions older than SESSION_TTL_HOURS are pruned on the periodic sweep.
"""
import asyncio
import json
import os
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

import config

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str           # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    messages: List[Message] = field(default_factory=list)
    compact_summary: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")

    def format_history(self) -> str:
        parts = []
        if self.compact_summary:
            parts.append(f"[ملخص المحادثة السابقة]: {self.compact_summary}")
        for msg in self.messages:
            role_label = "المستخدم" if msg.role == "user" else "نور"
            parts.append(f"{role_label}: {msg.content}")
        return "\n".join(parts) if parts else "لا يوجد سجل محادثة سابق."

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        self.last_active = time.time()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "messages": [asdict(m) for m in self.messages],
            "compact_summary": self.compact_summary,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            session_id=data["session_id"],
            messages=[Message(**m) for m in data.get("messages", [])],
            compact_summary=data.get("compact_summary", ""),
            created_at=data.get("created_at", time.time()),
            last_active=data.get("last_active", time.time()),
        )


def _safe_filename(session_id: str) -> str:
    """Sanitize session_id for use as a filename — only allow alnum, dash, underscore."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in session_id)[:128]


class SessionManager:
    """Thread-safe session store with auto-compaction and optional disk persistence."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._pruner_task: Optional[asyncio.Task] = None

    # ── Persistence ─────────────────────────────────────────────────

    def _session_path(self, session_id: str) -> str:
        return os.path.join(config.SESSION_PERSIST_DIR, f"{_safe_filename(session_id)}.json")

    def _persist_sync(self, session: Session) -> None:
        """Write a session to disk. Caller must already hold the lock if needed."""
        if not config.SESSION_PERSIST:
            return
        try:
            os.makedirs(config.SESSION_PERSIST_DIR, exist_ok=True)
            tmp_path = self._session_path(session.session_id) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False)
            os.replace(tmp_path, self._session_path(session.session_id))
        except Exception as e:
            logger.warning(f"Failed to persist session {session.session_id}: {e}")

    def load_from_disk(self) -> int:
        """Load all persisted sessions from disk. Call once at startup. Returns count loaded."""
        if not config.SESSION_PERSIST or not os.path.isdir(config.SESSION_PERSIST_DIR):
            return 0
        loaded = 0
        for fname in os.listdir(config.SESSION_PERSIST_DIR):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(config.SESSION_PERSIST_DIR, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                sess = Session.from_dict(data)
                self._sessions[sess.session_id] = sess
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load session from {fname}: {e}")
        if loaded:
            logger.info(f"📂 Restored {loaded} session(s) from {config.SESSION_PERSIST_DIR}")
        return loaded

    # ── Pruning ─────────────────────────────────────────────────────

    async def prune_expired(self) -> int:
        """Remove sessions idle longer than SESSION_TTL_HOURS. Returns count pruned."""
        cutoff = time.time() - config.SESSION_TTL_HOURS * 3600
        removed = []
        async with self._lock:
            for sid, sess in list(self._sessions.items()):
                if sess.last_active < cutoff:
                    removed.append(sid)
                    del self._sessions[sid]
                    try:
                        path = self._session_path(sid)
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception as e:
                        logger.warning(f"Failed to remove on-disk session {sid}: {e}")
        if removed:
            logger.info(f"🧹 Pruned {len(removed)} expired session(s)")
        return len(removed)

    def start_pruner(self, interval_s: int = 3600):
        """Start a background task that prunes expired sessions every `interval_s` seconds."""
        if self._pruner_task is not None and not self._pruner_task.done():
            return

        async def _loop():
            while True:
                try:
                    await self.prune_expired()
                except Exception as e:
                    logger.warning(f"Session pruner errored: {e}")
                await asyncio.sleep(interval_s)

        self._pruner_task = asyncio.create_task(_loop())

    # ── Core API ────────────────────────────────────────────────────

    async def get_or_create(self, session_id: str) -> Session:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
                logger.info(f"📝 New session created: {session_id}")
            return self._sessions[session_id]

    async def add_user_message(self, session_id: str, content: str) -> Session:
        session = await self.get_or_create(session_id)
        async with self._lock:
            session.add_message("user", content)
            self._persist_sync(session)
        return session

    async def add_assistant_message(self, session_id: str, content: str) -> Session:
        session = await self.get_or_create(session_id)
        async with self._lock:
            session.add_message("assistant", content)
            self._persist_sync(session)
        return session

    async def should_compact(self, session_id: str) -> bool:
        session = await self.get_or_create(session_id)
        return session.turn_count > config.SESSION_MAX_TURNS

    async def compact(self, session_id: str, llm_call_fn) -> str:
        """Compact old messages into a summary. `llm_call_fn`: async callable(prompt, system_msg) -> str."""
        session = await self.get_or_create(session_id)

        async with self._lock:
            if session.turn_count <= config.SESSION_MAX_TURNS:
                return session.compact_summary

            keep_count = config.SESSION_KEEP_RECENT * 2  # user + assistant pairs
            old_messages = session.messages[:-keep_count] if keep_count < len(session.messages) else []
            recent_messages = session.messages[-keep_count:] if keep_count < len(session.messages) else session.messages

            if not old_messages:
                return session.compact_summary

            conversation_text = ""
            if session.compact_summary:
                conversation_text += f"[ملخص سابق]: {session.compact_summary}\n\n"
            for msg in old_messages:
                role_label = "المستخدم" if msg.role == "user" else "نور"
                conversation_text += f"{role_label}: {msg.content}\n"

        # Call LLM outside the lock — compaction can take seconds.
        from app.core.prompts import PROMPTS, SYSTEM_MESSAGES
        prompt = PROMPTS["compact_history"].format(conversation=conversation_text)
        summary = await llm_call_fn(prompt, SYSTEM_MESSAGES["compact"])

        async with self._lock:
            session.compact_summary = summary
            session.messages = recent_messages
            self._persist_sync(session)
            logger.info(
                f"🗜️ Session {session_id} compacted: "
                f"{len(old_messages)} old messages → summary, keeping {len(recent_messages)} recent"
            )

        return summary

    async def delete_session(self, session_id: str) -> bool:
        async with self._lock:
            if session_id not in self._sessions:
                return False
            del self._sessions[session_id]
            try:
                path = self._session_path(session_id)
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to remove session file {session_id}: {e}")
            logger.info(f"🗑️ Session deleted: {session_id}")
            return True

    async def get_session_info(self, session_id: str) -> Optional[dict]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            return {
                "session_id": session.session_id,
                "turn_count": session.turn_count,
                "message_count": len(session.messages),
                "has_summary": bool(session.compact_summary),
                "created_at": session.created_at,
                "last_active": session.last_active,
            }


# ── Global singleton ──
session_manager = SessionManager()
