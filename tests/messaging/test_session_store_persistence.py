"""Persistence- und Concurrency-Tests fuer SessionStore (Public-API only).

Diese Tests greifen ausschliesslich ueber die public API auf SessionStore zu,
damit sie nach dem geplanten SQLite-Refactor unveraendert weiter passen.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from messaging.session import SessionStore


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "sessions.json")


@pytest.fixture
def fresh_store(store_path):
    return SessionStore(storage_path=store_path)


def _tree(root_id: str, *node_ids: str) -> dict:
    return {
        "root_id": root_id,
        "nodes": {nid: {"node_id": nid} for nid in (root_id, *node_ids)},
    }


class TestRoundtrip:
    def test_save_then_load_in_new_instance(self, store_path):
        s1 = SessionStore(storage_path=store_path)
        s1.save_tree("r1", _tree("r1", "n1", "n2"))
        s1.save_tree("r2", _tree("r2"))
        s1.flush_pending_save()

        s2 = SessionStore(storage_path=store_path)
        assert s2.get_tree("r1") == _tree("r1", "n1", "n2")
        assert s2.get_tree("r2") == _tree("r2")
        # Node mappings are reconstructed.
        mapping = s2.get_node_mapping()
        assert mapping["n1"] == "r1"
        assert mapping["n2"] == "r1"

    def test_remove_tree_clears_node_mappings(self, fresh_store):
        fresh_store.save_tree("r1", _tree("r1", "a", "b"))
        fresh_store.remove_tree("r1")
        assert fresh_store.get_tree("r1") is None
        assert "a" not in fresh_store.get_node_mapping()

    def test_register_and_remove_node_mappings(self, fresh_store):
        fresh_store.save_tree("r1", _tree("r1"))
        fresh_store.register_node("extra1", "r1")
        fresh_store.register_node("extra2", "r1")
        assert fresh_store.get_node_mapping()["extra1"] == "r1"

        fresh_store.remove_node_mappings(["extra1", "missing"])
        mapping = fresh_store.get_node_mapping()
        assert "extra1" not in mapping
        assert "extra2" in mapping  # untouched

    def test_get_all_trees_returns_copy(self, fresh_store):
        fresh_store.save_tree("r1", _tree("r1"))
        snapshot = fresh_store.get_all_trees()
        snapshot["r2"] = _tree("r2")  # mutating copy must not leak back
        assert "r2" not in fresh_store.get_all_trees()


class TestMessageLog:
    def test_record_and_retrieve_in_order(self, fresh_store):
        for mid in ("a", "b", "c"):
            fresh_store.record_message_id("telegram", "chat1", mid, "in", "content")
        assert fresh_store.get_message_ids_for_chat("telegram", "chat1") == [
            "a",
            "b",
            "c",
        ]

    def test_dedup_same_id(self, fresh_store):
        fresh_store.record_message_id("telegram", "chat1", "x", "in", "content")
        fresh_store.record_message_id("telegram", "chat1", "x", "out", "content")
        assert fresh_store.get_message_ids_for_chat("telegram", "chat1") == ["x"]

    def test_per_chat_isolation(self, fresh_store):
        fresh_store.record_message_id("telegram", "chat1", "1", "in", "content")
        fresh_store.record_message_id("telegram", "chat2", "2", "in", "content")
        assert fresh_store.get_message_ids_for_chat("telegram", "chat1") == ["1"]
        assert fresh_store.get_message_ids_for_chat("telegram", "chat2") == ["2"]

    def test_per_platform_isolation(self, fresh_store):
        fresh_store.record_message_id("telegram", "c", "1", "in", "content")
        fresh_store.record_message_id("discord", "c", "2", "in", "content")
        assert fresh_store.get_message_ids_for_chat("telegram", "c") == ["1"]
        assert fresh_store.get_message_ids_for_chat("discord", "c") == ["2"]

    def test_persistence_roundtrip(self, store_path):
        s1 = SessionStore(storage_path=store_path)
        for mid in ("1", "2", "3"):
            s1.record_message_id("telegram", "c1", mid, "in", "content")
        s1.flush_pending_save()

        s2 = SessionStore(storage_path=store_path)
        assert s2.get_message_ids_for_chat("telegram", "c1") == ["1", "2", "3"]


class TestClearAll:
    def test_clear_all_persists_empty_state(self, store_path):
        s1 = SessionStore(storage_path=store_path)
        s1.save_tree("r1", _tree("r1", "n1"))
        s1.record_message_id("telegram", "c1", "m1", "in", "content")
        s1.clear_all()

        assert s1.get_all_trees() == {}
        assert s1.get_node_mapping() == {}
        assert s1.get_message_ids_for_chat("telegram", "c1") == []

        s2 = SessionStore(storage_path=store_path)
        assert s2.get_all_trees() == {}
        assert s2.get_message_ids_for_chat("telegram", "c1") == []


class TestFlushPendingSave:
    def test_flush_persists_immediately(self, store_path):
        s1 = SessionStore(storage_path=store_path)
        s1.save_tree("r1", _tree("r1"))
        s1.flush_pending_save()  # Should be a hard sync.

        # New instance must see the data without waiting on a debounce timer.
        s2 = SessionStore(storage_path=store_path)
        assert s2.get_tree("r1") == _tree("r1")

    def test_flush_is_idempotent(self, fresh_store):
        fresh_store.save_tree("r1", _tree("r1"))
        fresh_store.flush_pending_save()
        fresh_store.flush_pending_save()  # second call must not raise
        fresh_store.flush_pending_save()


class TestConcurrency:
    """Concurrent writes via ThreadPoolExecutor must remain consistent."""

    def test_concurrent_save_tree(self, fresh_store):
        def worker(idx: int) -> None:
            fresh_store.save_tree(f"r{idx}", _tree(f"r{idx}", f"n{idx}"))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(50)))

        fresh_store.flush_pending_save()
        for idx in range(50):
            assert fresh_store.get_tree(f"r{idx}") is not None

    def test_concurrent_record_message_id(self, fresh_store):
        def worker(idx: int) -> None:
            fresh_store.record_message_id(
                "telegram", "chat1", f"m{idx}", "in", "content"
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(100)))

        fresh_store.flush_pending_save()
        ids = fresh_store.get_message_ids_for_chat("telegram", "chat1")
        # Order isn't guaranteed across threads, but completeness and no dupes are.
        assert sorted(ids) == sorted(f"m{i}" for i in range(100))
        assert len(ids) == len(set(ids))

    def test_mixed_concurrent_operations(self, fresh_store):
        errors: list[Exception] = []
        lock = threading.Lock()

        def writer():
            try:
                for i in range(20):
                    fresh_store.save_tree(
                        f"thread_r_{threading.get_ident()}_{i}",
                        _tree(f"thread_r_{threading.get_ident()}_{i}"),
                    )
            except Exception as exc:
                with lock:
                    errors.append(exc)

        def recorder():
            try:
                for i in range(20):
                    fresh_store.record_message_id(
                        "telegram",
                        "chat1",
                        f"t{threading.get_ident()}_{i}",
                        "in",
                        "content",
                    )
            except Exception as exc:
                with lock:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(writer) for _ in range(3)] + [
                pool.submit(recorder) for _ in range(3)
            ]
            for f in futures:
                f.result(timeout=10)

        # Allow any debounced save to fire so persistence is consistent.
        fresh_store.flush_pending_save()
        assert errors == [], f"Concurrent ops raised: {errors}"

    def test_concurrent_save_and_remove(self, fresh_store):
        for i in range(30):
            fresh_store.save_tree(f"r{i}", _tree(f"r{i}"))

        def remover(i: int) -> None:
            fresh_store.remove_tree(f"r{i}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(remover, range(15)))

        for i in range(15):
            assert fresh_store.get_tree(f"r{i}") is None
        for i in range(15, 30):
            assert fresh_store.get_tree(f"r{i}") is not None


class TestSyncFromTreeData:
    """sync_from_tree_data atomically replaces internal state."""

    def test_sync_replaces_state(self, fresh_store):
        fresh_store.save_tree("old", _tree("old"))
        new_trees = {"new1": _tree("new1", "n1")}
        new_mapping = {"new1": "new1", "n1": "new1"}
        fresh_store.sync_from_tree_data(new_trees, new_mapping)

        assert fresh_store.get_tree("old") is None
        assert fresh_store.get_tree("new1") == _tree("new1", "n1")
        assert fresh_store.get_node_mapping() == new_mapping

    def test_sync_persists(self, store_path):
        s1 = SessionStore(storage_path=store_path)
        s1.sync_from_tree_data({"r1": _tree("r1")}, {"r1": "r1"})
        s1.flush_pending_save()

        s2 = SessionStore(storage_path=store_path)
        assert s2.get_tree("r1") == _tree("r1")


class TestDebouncedSave:
    """Loose timing test: flush_pending_save must complete a pending save."""

    def test_save_eventually_visible_to_new_instance(self, store_path):
        s1 = SessionStore(storage_path=store_path)
        s1.save_tree("r1", _tree("r1"))
        # Allow debounce to fire if applicable, then flush as belt-and-braces.
        time.sleep(0.7)
        s1.flush_pending_save()

        s2 = SessionStore(storage_path=store_path)
        assert s2.get_tree("r1") == _tree("r1")
