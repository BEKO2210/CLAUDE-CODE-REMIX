"""
Session Store for Messaging Platforms (SQLite backend).

Provides persistent storage for mapping platform messages to Claude CLI session IDs
and message trees for conversation continuation.

Storage backend: SQLite with WAL mode for crash-safe concurrent access.

Backwards compatibility: when constructed with a legacy `sessions.json` path,
the store auto-migrates the JSON file into a sibling `sessions.sqlite` once,
keeping the original `sessions.json` (renamed to `sessions.json.bak`) as a
backup.
"""

import contextlib
import json
import os
import shutil
import sqlite3
import threading
import time
from datetime import UTC, datetime

from loguru import logger


def _resolve_db_path(storage_path: str) -> str:
    """Map a constructor `storage_path` to the actual SQLite db file.

    `.json` paths are treated as legacy: the database lives in a sibling
    `.sqlite` file with the same stem. Any other extension is used verbatim.
    """
    if storage_path.endswith(".json"):
        return storage_path[: -len(".json")] + ".sqlite"
    return storage_path


class SessionStore:
    """
    Persistent SQLite-backed storage for message ↔ Claude session mappings
    and message trees. Public API is identical to the previous JSON store.
    """

    def __init__(self, storage_path: str = "sessions.json"):
        self.storage_path = storage_path
        self._db_path = _resolve_db_path(storage_path)
        self._lock = threading.Lock()
        cap_raw = os.getenv("MAX_MESSAGE_LOG_ENTRIES_PER_CHAT", "").strip()
        try:
            self._message_log_cap: int | None = int(cap_raw) if cap_raw else None
        except ValueError:
            self._message_log_cap = None

        self._migrate_legacy_json_if_needed()
        self._closed = False
        self._conn: sqlite3.Connection = sqlite3.connect(
            self._db_path, check_same_thread=False, isolation_level=None
        )
        self._configure_connection(self._conn)
        self._init_schema(self._conn)
        self._log_load_summary()

    # ------------------------------------------------------------------
    # Connection / schema setup
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_connection(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trees (
                root_id    TEXT PRIMARY KEY,
                data       TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS node_to_tree (
                node_id TEXT PRIMARY KEY,
                root_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS message_log (
                chat_key   TEXT NOT NULL,
                message_id TEXT NOT NULL,
                ts         TEXT NOT NULL,
                direction  TEXT NOT NULL,
                kind       TEXT NOT NULL,
                insert_ord INTEGER NOT NULL,
                PRIMARY KEY (chat_key, message_id)
            );
            CREATE INDEX IF NOT EXISTS idx_msg_log_chat
                ON message_log(chat_key, insert_ord);
            CREATE INDEX IF NOT EXISTS idx_node_to_tree_root
                ON node_to_tree(root_id);
            """
        )

    def _log_load_summary(self) -> None:
        with self._lock:
            tree_count = self._conn.execute("SELECT COUNT(*) FROM trees").fetchone()[0]
            msg_count = self._conn.execute(
                "SELECT COUNT(*) FROM message_log"
            ).fetchone()[0]
        logger.info(
            f"SessionStore ready: {tree_count} trees, "
            f"{msg_count} msg_ids ({self._db_path})"
        )

    # ------------------------------------------------------------------
    # Migration from legacy JSON storage
    # ------------------------------------------------------------------

    def _migrate_legacy_json_if_needed(self) -> None:
        """If a legacy sessions.json exists and no sqlite sibling yet, migrate."""
        json_path = self.storage_path
        if not json_path.endswith(".json"):
            return
        if not os.path.exists(json_path):
            return
        if os.path.exists(self._db_path):
            return  # already migrated

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Legacy session migration: failed to read {json_path}: {e}")
            return

        if not isinstance(data, dict):
            logger.warning(
                f"Legacy session migration: unexpected JSON shape in {json_path}"
            )
            return

        tmp_path = self._db_path + ".tmp"
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            tmp_conn = sqlite3.connect(tmp_path, isolation_level=None)
            try:
                self._configure_connection(tmp_conn)
                self._init_schema(tmp_conn)
                self._migrate_load_into(tmp_conn, data)
            finally:
                tmp_conn.close()
            os.replace(tmp_path, self._db_path)
        except (OSError, sqlite3.Error) as e:
            logger.error(f"Legacy session migration failed: {e}")
            if os.path.exists(tmp_path):
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
            return

        try:
            shutil.copy2(json_path, json_path + ".bak")
        except OSError as e:
            logger.warning(f"Failed to back up legacy {json_path}: {e}")

        logger.info(
            f"Migrated legacy sessions.json → {self._db_path} "
            f"(backup at {json_path}.bak)"
        )

    @staticmethod
    def _migrate_load_into(conn: sqlite3.Connection, data: dict) -> None:
        trees = data.get("trees") or {}
        node_to_tree = data.get("node_to_tree") or {}
        message_log = data.get("message_log") or {}

        now = time.time()
        conn.execute("BEGIN IMMEDIATE")
        try:
            if isinstance(trees, dict):
                for root_id, tree_data in trees.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO trees(root_id, data, updated_at) "
                        "VALUES (?, ?, ?)",
                        (str(root_id), json.dumps(tree_data), now),
                    )
            if isinstance(node_to_tree, dict):
                for node_id, root_id in node_to_tree.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO node_to_tree(node_id, root_id) "
                        "VALUES (?, ?)",
                        (str(node_id), str(root_id)),
                    )
            if isinstance(message_log, dict):
                ord_counter = 0
                for chat_key, items in message_log.items():
                    if not isinstance(chat_key, str) or not isinstance(items, list):
                        continue
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        mid = item.get("message_id")
                        if mid is None:
                            continue
                        ord_counter += 1
                        conn.execute(
                            "INSERT OR IGNORE INTO message_log"
                            "(chat_key, message_id, ts, direction, kind, insert_ord) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                chat_key,
                                str(mid),
                                str(item.get("ts") or ""),
                                str(item.get("direction") or ""),
                                str(item.get("kind") or ""),
                                ord_counter,
                            ),
                        )
            conn.execute("COMMIT")
        except sqlite3.Error:
            conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_chat_key(self, platform: str, chat_id: str) -> str:
        return f"{platform}:{chat_id}"

    def flush_pending_save(self) -> None:
        """No-op: SQLite commits each write synchronously.

        Kept for backwards compatibility with callers that flushed the
        debounced JSON store on shutdown.
        """
        return

    # ------------------------------------------------------------------
    # Message log
    # ------------------------------------------------------------------

    def record_message_id(
        self,
        platform: str,
        chat_id: str,
        message_id: str,
        direction: str,
        kind: str,
    ) -> None:
        """Record a message_id for later best-effort deletion (/clear)."""
        if message_id is None:
            return

        chat_key = self._make_chat_key(str(platform), str(chat_id))
        mid = str(message_id)
        ts = datetime.now(UTC).isoformat()

        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM message_log WHERE chat_key = ? AND message_id = ?",
                (chat_key, mid),
            )
            if cur.fetchone() is not None:
                return

            next_ord = self._conn.execute(
                "SELECT COALESCE(MAX(insert_ord), 0) + 1 FROM message_log "
                "WHERE chat_key = ?",
                (chat_key,),
            ).fetchone()[0]

            self._conn.execute(
                "INSERT INTO message_log"
                "(chat_key, message_id, ts, direction, kind, insert_ord) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chat_key, mid, ts, str(direction), str(kind), next_ord),
            )

            cap = self._message_log_cap
            if cap is not None and cap > 0:
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM message_log WHERE chat_key = ?",
                    (chat_key,),
                ).fetchone()[0]
                if count > cap:
                    self._conn.execute(
                        "DELETE FROM message_log "
                        "WHERE chat_key = ? AND insert_ord IN ("
                        "  SELECT insert_ord FROM message_log "
                        "  WHERE chat_key = ? "
                        "  ORDER BY insert_ord ASC LIMIT ?"
                        ")",
                        (chat_key, chat_key, count - cap),
                    )

    def get_message_ids_for_chat(self, platform: str, chat_id: str) -> list[str]:
        """Get all recorded message IDs for a chat (in insertion order)."""
        chat_key = self._make_chat_key(str(platform), str(chat_id))
        with self._lock:
            rows = self._conn.execute(
                "SELECT message_id FROM message_log "
                "WHERE chat_key = ? ORDER BY insert_ord ASC",
                (chat_key,),
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Wipe
    # ------------------------------------------------------------------

    def clear_all(self) -> None:
        """Remove all trees, node mappings and message logs."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM trees")
                self._conn.execute("DELETE FROM node_to_tree")
                self._conn.execute("DELETE FROM message_log")
                self._conn.execute("COMMIT")
            except sqlite3.Error:
                self._conn.execute("ROLLBACK")
                raise

    # ==================== Tree Methods ====================

    def save_tree(self, root_id: str, tree_data: dict) -> None:
        """Save a message tree and reconcile node→root mappings for it."""
        root_id_s = str(root_id)
        payload = json.dumps(tree_data)
        nodes = tree_data.get("nodes") if isinstance(tree_data, dict) else {}
        if not isinstance(nodes, dict):
            nodes = {}
        node_ids = list(nodes.keys())

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO trees(root_id, data, updated_at) "
                    "VALUES (?, ?, ?)",
                    (root_id_s, payload, time.time()),
                )
                for node_id in node_ids:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO node_to_tree(node_id, root_id) "
                        "VALUES (?, ?)",
                        (str(node_id), root_id_s),
                    )
                self._conn.execute("COMMIT")
            except sqlite3.Error:
                self._conn.execute("ROLLBACK")
                raise
        logger.debug(f"Saved tree {root_id_s}")

    def get_tree(self, root_id: str) -> dict | None:
        """Get a tree by its root ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM trees WHERE root_id = ?", (str(root_id),)
            ).fetchone()
        if row is None:
            return None
        try:
            parsed = json.loads(row[0])
        except json.JSONDecodeError:
            logger.warning(f"Corrupt tree payload for {root_id}; treating as missing")
            return None
        return parsed if isinstance(parsed, dict) else None

    def register_node(self, node_id: str, root_id: str) -> None:
        """Register a node ID to a tree root."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO node_to_tree(node_id, root_id) VALUES (?, ?)",
                (str(node_id), str(root_id)),
            )

    def remove_node_mappings(self, node_ids: list[str]) -> None:
        """Remove node IDs from the node-to-tree mapping."""
        if not node_ids:
            return
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.executemany(
                    "DELETE FROM node_to_tree WHERE node_id = ?",
                    [(str(nid),) for nid in node_ids],
                )
                self._conn.execute("COMMIT")
            except sqlite3.Error:
                self._conn.execute("ROLLBACK")
                raise

    def remove_tree(self, root_id: str) -> None:
        """Remove a tree and all its node mappings from the store."""
        root_id_s = str(root_id)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT data FROM trees WHERE root_id = ?", (root_id_s,)
                ).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return

                node_ids: list[str] = []
                try:
                    tree_data = json.loads(row[0])
                except json.JSONDecodeError:
                    tree_data = None
                if isinstance(tree_data, dict):
                    nodes = tree_data.get("nodes")
                    if isinstance(nodes, dict):
                        node_ids = [str(n) for n in nodes]

                self._conn.execute("DELETE FROM trees WHERE root_id = ?", (root_id_s,))
                if node_ids:
                    self._conn.executemany(
                        "DELETE FROM node_to_tree WHERE node_id = ?",
                        [(nid,) for nid in node_ids],
                    )
                self._conn.execute("COMMIT")
            except sqlite3.Error:
                self._conn.execute("ROLLBACK")
                raise

    def get_all_trees(self) -> dict[str, dict]:
        """Get all stored trees (public accessor)."""
        with self._lock:
            rows = self._conn.execute("SELECT root_id, data FROM trees").fetchall()

        result: dict[str, dict] = {}
        for root_id, payload in rows:
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                result[root_id] = parsed
        return result

    def get_node_mapping(self) -> dict[str, str]:
        """Get the node-to-tree mapping (public accessor)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT node_id, root_id FROM node_to_tree"
            ).fetchall()
        return dict(rows)

    def sync_from_tree_data(
        self, trees: dict[str, dict], node_to_tree: dict[str, str]
    ) -> None:
        """Replace internal tree state atomically with the given snapshot."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM trees")
                self._conn.execute("DELETE FROM node_to_tree")
                now = time.time()
                for root_id, tree_data in trees.items():
                    self._conn.execute(
                        "INSERT INTO trees(root_id, data, updated_at) VALUES (?, ?, ?)",
                        (str(root_id), json.dumps(tree_data), now),
                    )
                for node_id, root_id in node_to_tree.items():
                    self._conn.execute(
                        "INSERT OR REPLACE INTO node_to_tree(node_id, root_id) "
                        "VALUES (?, ?)",
                        (str(node_id), str(root_id)),
                    )
                self._conn.execute("COMMIT")
            except sqlite3.Error:
                self._conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection (idempotent)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.close()
            except sqlite3.Error as e:
                logger.warning(f"Error closing SessionStore connection: {e}")

    def __del__(self) -> None:
        # Destructor must never raise — connection cleanup is best-effort.
        with contextlib.suppress(Exception):
            self.close()
