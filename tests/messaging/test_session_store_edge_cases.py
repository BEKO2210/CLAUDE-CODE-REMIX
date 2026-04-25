"""Edge case tests for messaging/session.py SessionStore.

Backed by SQLite; legacy JSON files are auto-migrated on first construction.
"""

import json

import pytest

from messaging.session import SessionStore


@pytest.fixture
def tmp_store(tmp_path):
    """Create a SessionStore using a temp file."""
    path = str(tmp_path / "sessions.json")
    return SessionStore(storage_path=path)


class TestSessionStoreLoadEdgeCases:
    """Tests for loading corrupted/malformed legacy JSON during migration."""

    def test_load_corrupted_json(self, tmp_path):
        """Corrupted legacy JSON file is handled gracefully (logs error, starts empty)."""
        path = str(tmp_path / "sessions.json")
        with open(path, "w") as f:
            f.write("{invalid json")

        store = SessionStore(storage_path=path)
        assert store.get_all_trees() == {}

    def test_load_truncated_json(self, tmp_path):
        """Truncated legacy JSON file is handled gracefully."""
        path = str(tmp_path / "sessions.json")
        with open(path, "w") as f:
            f.write('{"sessions": {"s1": {"session_id": "s1"')

        store = SessionStore(storage_path=path)
        assert store.get_all_trees() == {}

    def test_load_empty_file(self, tmp_path):
        """Empty legacy JSON file is handled gracefully."""
        path = str(tmp_path / "sessions.json")
        with open(path, "w") as f:
            f.write("")

        store = SessionStore(storage_path=path)
        assert store.get_all_trees() == {}

    def test_load_nonexistent_file(self, tmp_path):
        """Non-existent legacy file starts with empty SQLite state."""
        path = str(tmp_path / "nonexistent.json")
        store = SessionStore(storage_path=path)
        assert store.get_all_trees() == {}

    def test_load_legacy_sessions_ignored(self, tmp_path):
        """Legacy `sessions` field is ignored; trees and message_log migrate."""
        path = str(tmp_path / "sessions.json")
        data = {
            "sessions": {
                "s1": {
                    "session_id": "s1",
                    "chat_id": 12345,
                    "initial_msg_id": 100,
                    "last_msg_id": 200,
                    "platform": "telegram",
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "updated_at": "2025-01-01T00:00:00+00:00",
                }
            },
            "trees": {"r1": {"root_id": "r1", "nodes": {"r1": {}}}},
            "node_to_tree": {"r1": "r1"},
            "message_log": {},
        }
        with open(path, "w") as f:
            json.dump(data, f)

        store = SessionStore(storage_path=path)
        assert store.get_tree("r1") is not None
        assert store.get_node_mapping() == {"r1": "r1"}


class TestSessionStoreLifecycle:
    """Tests for connection lifecycle / close behaviour."""

    def test_close_is_idempotent(self, tmp_store):
        tmp_store.save_tree("r1", {"root_id": "r1", "nodes": {"r1": {}}})
        tmp_store.close()
        tmp_store.close()  # second close must not raise


class TestSessionStoreClearAll:
    def test_clear_all_wipes_state_and_persists(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)

        store.save_tree(
            "root1",
            {
                "root_id": "root1",
                "nodes": {
                    "root1": {
                        "node_id": "root1",
                        "incoming": {
                            "text": "hello",
                            "chat_id": "c1",
                            "user_id": "u1",
                            "message_id": "m1",
                            "platform": "telegram",
                            "reply_to_message_id": None,
                            "username": None,
                        },
                        "status_message_id": "status1",
                        "state": "pending",
                        "parent_id": None,
                        "session_id": None,
                        "children_ids": [],
                        "created_at": "2025-01-01T00:00:00+00:00",
                        "completed_at": None,
                        "error_message": None,
                    }
                },
            },
        )

        store.clear_all()

        assert store.get_all_trees() == {}
        assert store.get_node_mapping() == {}

        # Re-open: state must remain empty.
        store.close()
        store2 = SessionStore(storage_path=path)
        assert store2.get_all_trees() == {}

    def test_message_log_persists_and_dedups(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)

        store.record_message_id("telegram", "c1", "1", direction="in", kind="command")
        store.record_message_id("telegram", "c1", "2", direction="out", kind="command")
        store.record_message_id("telegram", "c1", "2", direction="out", kind="command")

        ids = store.get_message_ids_for_chat("telegram", "c1")
        assert ids == ["1", "2"]

        store.flush_pending_save()
        store.close()
        store2 = SessionStore(storage_path=path)
        assert store2.get_message_ids_for_chat("telegram", "c1") == ["1", "2"]
