"""SQLite-spezifische Tests fuer SessionStore (Migration, WAL, Crash-Safety)."""

import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from messaging.session import SessionStore, _resolve_db_path


def _legacy_payload() -> dict:
    return {
        "trees": {
            "r1": {"root_id": "r1", "nodes": {"r1": {}, "n1": {}}},
            "r2": {"root_id": "r2", "nodes": {"r2": {}}},
        },
        "node_to_tree": {"r1": "r1", "n1": "r1", "r2": "r2"},
        "message_log": {
            "telegram:c1": [
                {"message_id": "1", "ts": "t1", "direction": "in", "kind": "content"},
                {"message_id": "2", "ts": "t2", "direction": "out", "kind": "status"},
            ]
        },
        # Legacy field that must be ignored.
        "sessions": {"s1": {"session_id": "s1"}},
    }


class TestPathResolution:
    def test_json_path_maps_to_sqlite(self, tmp_path):
        json_path = str(tmp_path / "sessions.json")
        assert _resolve_db_path(json_path).endswith("sessions.sqlite")

    def test_non_json_path_used_verbatim(self, tmp_path):
        db_path = str(tmp_path / "store.db")
        assert _resolve_db_path(db_path) == db_path


class TestLegacyMigration:
    def test_migration_imports_trees_and_log(self, tmp_path):
        json_path = str(tmp_path / "sessions.json")
        with open(json_path, "w") as f:
            json.dump(_legacy_payload(), f)

        store = SessionStore(storage_path=json_path)

        assert store.get_tree("r1") == {
            "root_id": "r1",
            "nodes": {"r1": {}, "n1": {}},
        }
        assert store.get_tree("r2") == {"root_id": "r2", "nodes": {"r2": {}}}
        mapping = store.get_node_mapping()
        assert mapping["r1"] == "r1"
        assert mapping["n1"] == "r1"
        assert mapping["r2"] == "r2"
        assert store.get_message_ids_for_chat("telegram", "c1") == ["1", "2"]

    def test_migration_creates_backup(self, tmp_path):
        json_path = str(tmp_path / "sessions.json")
        with open(json_path, "w") as f:
            json.dump(_legacy_payload(), f)

        SessionStore(storage_path=json_path)

        assert os.path.exists(json_path + ".bak"), "legacy JSON backup missing"
        assert os.path.exists(str(tmp_path / "sessions.sqlite"))

    def test_migration_skipped_when_sqlite_exists(self, tmp_path):
        json_path = str(tmp_path / "sessions.json")
        sqlite_path = str(tmp_path / "sessions.sqlite")

        # Pre-create both: legacy JSON with one tree, but a SQLite file
        # that has different (empty) content.
        with open(json_path, "w") as f:
            json.dump(_legacy_payload(), f)
        # Touch a valid empty SQLite file.
        conn = sqlite3.connect(sqlite_path)
        conn.close()

        store = SessionStore(storage_path=json_path)
        # Schema is created, but legacy data is NOT imported because the
        # sqlite sibling pre-existed.
        assert store.get_all_trees() == {}
        # Backup must NOT be created because we skipped migration.
        assert not os.path.exists(json_path + ".bak")

    def test_migration_handles_corrupt_legacy_gracefully(self, tmp_path):
        json_path = str(tmp_path / "sessions.json")
        with open(json_path, "w") as f:
            f.write("{not valid json")

        store = SessionStore(storage_path=json_path)
        assert store.get_all_trees() == {}
        assert store.get_node_mapping() == {}


class TestSqliteSemantics:
    def test_wal_mode_enabled(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        SessionStore(storage_path=path)

        sqlite_path = str(tmp_path / "sessions.sqlite")
        conn = sqlite3.connect(sqlite_path)
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()
        assert mode.lower() == "wal"

    def test_save_visible_to_independent_reader(self, tmp_path):
        """A second connection (raw sqlite3) must see committed writes."""
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)
        store.save_tree("r1", {"root_id": "r1", "nodes": {"r1": {}}})

        sqlite_path = str(tmp_path / "sessions.sqlite")
        reader = sqlite3.connect(sqlite_path)
        try:
            row = reader.execute(
                "SELECT root_id FROM trees WHERE root_id = 'r1'"
            ).fetchone()
        finally:
            reader.close()
        assert row == ("r1",)

    def test_reopen_after_close_preserves_data(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)
        store.save_tree("rx", {"root_id": "rx", "nodes": {"rx": {}, "ny": {}}})
        store.record_message_id("telegram", "c1", "m1", "in", "content")
        store.close()

        store2 = SessionStore(storage_path=path)
        assert store2.get_tree("rx") == {
            "root_id": "rx",
            "nodes": {"rx": {}, "ny": {}},
        }
        assert store2.get_message_ids_for_chat("telegram", "c1") == ["m1"]

    def test_corrupt_tree_payload_returned_as_none(self, tmp_path):
        """A row with invalid JSON must not crash get_tree / get_all_trees."""
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)
        store.save_tree("good", {"root_id": "good", "nodes": {"good": {}}})

        # Corrupt the row directly.
        sqlite_path = str(tmp_path / "sessions.sqlite")
        conn = sqlite3.connect(sqlite_path, isolation_level=None)
        try:
            conn.execute(
                "INSERT INTO trees(root_id, data, updated_at) VALUES(?, ?, ?)",
                ("bad", "{not json", 0.0),
            )
        finally:
            conn.close()

        # Tree-level: corrupt payload returns None.
        assert store.get_tree("bad") is None
        # Aggregate: corrupt rows are silently filtered out.
        all_trees = store.get_all_trees()
        assert "bad" not in all_trees
        assert "good" in all_trees


class TestConcurrencyOnSqlite:
    """Threaded writes against a single SessionStore must remain consistent."""

    def test_parallel_save_tree(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)

        def worker(idx: int) -> None:
            store.save_tree(
                f"r{idx}",
                {"root_id": f"r{idx}", "nodes": {f"n{idx}": {}}},
            )

        with ThreadPoolExecutor(max_workers=10) as pool:
            list(pool.map(worker, range(50)))

        all_trees = store.get_all_trees()
        assert len(all_trees) == 50
        for idx in range(50):
            assert f"r{idx}" in all_trees

    def test_parallel_record_message_id_no_dupes(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)

        def worker(idx: int) -> None:
            store.record_message_id("telegram", "chat1", f"m{idx}", "in", "content")

        with ThreadPoolExecutor(max_workers=10) as pool:
            list(pool.map(worker, range(200)))

        ids = store.get_message_ids_for_chat("telegram", "chat1")
        assert len(ids) == 200
        assert len(set(ids)) == 200

    def test_parallel_mixed_operations(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)

        errors: list[Exception] = []
        lock = threading.Lock()

        def writer(i: int) -> None:
            try:
                store.save_tree(f"r{i}", {"root_id": f"r{i}", "nodes": {f"n{i}": {}}})
                store.register_node(f"extra_{i}", f"r{i}")
                if i % 5 == 0:
                    store.remove_tree(f"r{i}")
            except Exception as exc:
                with lock:
                    errors.append(exc)

        def recorder(i: int) -> None:
            try:
                store.record_message_id("telegram", "chat1", f"msg{i}", "in", "content")
            except Exception as exc:
                with lock:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(writer, i) for i in range(40)]
            futures += [pool.submit(recorder, i) for i in range(60)]
            for f in futures:
                f.result(timeout=10)

        assert errors == [], f"Concurrent operations raised: {errors}"
        # 40 saves, 8 removed by index%5 (0,5,10,15,20,25,30,35) → 32 trees left.
        assert len(store.get_all_trees()) == 32


class TestMessageLogCap:
    def test_cap_drops_oldest(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAX_MESSAGE_LOG_ENTRIES_PER_CHAT", "5")
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)

        for i in range(10):
            store.record_message_id("telegram", "c1", f"m{i}", "in", "content")

        ids = store.get_message_ids_for_chat("telegram", "c1")
        assert ids == [f"m{i}" for i in range(5, 10)]

    def test_no_cap_keeps_all(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)
        for i in range(15):
            store.record_message_id("telegram", "c1", f"m{i}", "in", "content")
        assert len(store.get_message_ids_for_chat("telegram", "c1")) == 15


class TestFlushAndClose:
    def test_flush_pending_save_is_noop(self, tmp_path):
        """Backwards-compat method: must exist and not raise."""
        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        store.save_tree("r1", {"root_id": "r1", "nodes": {"r1": {}}})
        # Must be safe to call repeatedly.
        store.flush_pending_save()
        store.flush_pending_save()
        # Data is already durable without flushing.
        store.close()
        store2 = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        assert store2.get_tree("r1") == {"root_id": "r1", "nodes": {"r1": {}}}

    def test_close_idempotent(self, tmp_path):
        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        store.close()
        store.close()  # second close must not raise


@pytest.mark.parametrize("ext", [".sqlite", ".db"])
def test_explicit_sqlite_path_skips_migration(tmp_path, ext):
    """When constructed with a non-.json path, no migration logic runs."""
    path = str(tmp_path / f"store{ext}")
    store = SessionStore(storage_path=path)
    store.save_tree("r1", {"root_id": "r1", "nodes": {"r1": {}}})
    store.close()

    # No legacy files were created or touched.
    assert not os.path.exists(str(tmp_path / "store.json"))
    assert not os.path.exists(str(tmp_path / "store.json.bak"))

    store2 = SessionStore(storage_path=path)
    assert store2.get_tree("r1") is not None
