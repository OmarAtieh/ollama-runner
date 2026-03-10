"""SQLite-backed session persistence for chat history."""

import time
import uuid

import aiosqlite

from app.config import SESSIONS_DIR


class _ConnectionContext:
    """Async context manager that opens an aiosqlite connection with foreign keys."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db = None

    async def __aenter__(self):
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA foreign_keys=ON")
        return self._db

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._db:
            await self._db.close()
        return False


class SessionManager:
    """Manages chat sessions and messages in a SQLite database."""

    def __init__(self):
        self._db_path = SESSIONS_DIR / "sessions.db"

    async def init_db(self) -> None:
        """Create tables if not exist and enable WAL mode."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    tokens_per_second REAL DEFAULT 0,
                    time_to_first_token_ms REAL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id)
                """
            )
            await db.commit()

    def _connect(self):
        """Return a new aiosqlite connection context manager with foreign keys enabled."""
        return _ConnectionContext(str(self._db_path))

    async def create_session(self, title: str, model_id: str | None = None) -> dict:
        """Create a new session with a short UUID (8 chars)."""
        session_id = uuid.uuid4().hex[:8]
        now = time.time()
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO sessions (id, title, model_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, title, model_id, now, now),
            )
            await db.commit()
        return {
            "id": session_id,
            "title": title,
            "model_id": model_id,
            "created_at": now,
            "updated_at": now,
        }

    async def list_sessions(self) -> list[dict]:
        """Return all sessions ordered by updated_at DESC."""
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT id, title, model_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            )
            rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "title": row[1],
                "model_id": row[2],
                "created_at": row[3],
                "updated_at": row[4],
            }
            for row in rows
        ]

    async def get_session(self, session_id: str) -> dict | None:
        """Get a single session by ID, or None if not found."""
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT id, title, model_id, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "model_id": row[2],
            "created_at": row[3],
            "updated_at": row[4],
        }

    async def update_session(
        self, session_id: str, title: str | None = None, model_id: str | None = None
    ) -> None:
        """Update session title and/or model_id."""
        now = time.time()
        async with self._connect() as db:
            if title is not None:
                await db.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (title, now, session_id),
                )
            if model_id is not None:
                await db.execute(
                    "UPDATE sessions SET model_id = ?, updated_at = ? WHERE id = ?",
                    (model_id, now, session_id),
                )
            await db.commit()

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and cascade-delete its messages."""
        async with self._connect() as db:
            # Manually delete messages first, then session (foreign_keys handles it
            # but explicit is safer across SQLite builds)
            await db.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await db.commit()

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        token_count: int = 0,
        tokens_per_second: float = 0,
        time_to_first_token_ms: float = 0,
    ) -> dict:
        """Add a message to a session."""
        now = time.time()
        async with self._connect() as db:
            cursor = await db.execute(
                """
                INSERT INTO messages
                    (session_id, role, content, timestamp, token_count, tokens_per_second, time_to_first_token_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, now, token_count, tokens_per_second, time_to_first_token_ms),
            )
            msg_id = cursor.lastrowid
            # Also bump session updated_at
            await db.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            await db.commit()
        return {
            "id": msg_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": now,
            "token_count": token_count,
            "tokens_per_second": tokens_per_second,
            "time_to_first_token_ms": time_to_first_token_ms,
        }

    async def get_messages(self, session_id: str) -> list[dict]:
        """Get all messages for a session, ordered by timestamp."""
        async with self._connect() as db:
            cursor = await db.execute(
                """
                SELECT id, session_id, role, content, timestamp,
                       token_count, tokens_per_second, time_to_first_token_ms
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "role": row[2],
                "content": row[3],
                "timestamp": row[4],
                "token_count": row[5],
                "tokens_per_second": row[6],
                "time_to_first_token_ms": row[7],
            }
            for row in rows
        ]

    async def get_session_token_count(self, session_id: str) -> int:
        """Sum of all message token_counts in a session."""
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT COALESCE(SUM(token_count), 0) FROM messages WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
        return int(row[0])
