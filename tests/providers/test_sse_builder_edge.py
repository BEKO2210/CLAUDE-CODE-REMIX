"""JSONDecodeError-Pfade in SSEBuilder Task-arg buffering.

Validiert die spezifischen Exception-Handler in providers/common/sse_builder.py
(buffer_task_args und flush_task_arg_buffers).
"""

import json

from providers.common.sse_builder import ContentBlockManager, ToolCallState


class TestBufferTaskArgs:
    """Tests for ContentBlockManager.buffer_task_args."""

    def test_partial_json_returns_none(self):
        cbm = ContentBlockManager()
        cbm.tool_states[0] = ToolCallState(
            block_index=0, tool_id="t1", name="Task", started=True
        )

        assert cbm.buffer_task_args(0, '{"description":') is None
        assert cbm.buffer_task_args(0, ' "do something"') is None
        result = cbm.buffer_task_args(0, "}")
        assert result == {"description": "do something", "run_in_background": False}

    def test_invalid_json_keeps_buffering(self):
        cbm = ContentBlockManager()
        cbm.tool_states[0] = ToolCallState(
            block_index=0, tool_id="t1", name="Task", started=True
        )
        # Invalid because `not_json` is not quoted; still parseable as fragment.
        assert cbm.buffer_task_args(0, "not_json") is None
        # State must not be marked emitted; buffer continues to accumulate.
        assert cbm.tool_states[0].task_args_emitted is False
        assert cbm.tool_states[0].task_arg_buffer == "not_json"

    def test_already_emitted_returns_none(self):
        cbm = ContentBlockManager()
        cbm.tool_states[0] = ToolCallState(
            block_index=0,
            tool_id="t1",
            name="Task",
            started=True,
            task_args_emitted=True,
        )
        assert cbm.buffer_task_args(0, '{"description": "x"}') is None

    def test_unknown_index_returns_none(self):
        cbm = ContentBlockManager()
        assert cbm.buffer_task_args(999, '{"a":1}') is None

    def test_run_in_background_forced_false(self):
        cbm = ContentBlockManager()
        cbm.tool_states[0] = ToolCallState(
            block_index=0, tool_id="t1", name="Task", started=True
        )
        result = cbm.buffer_task_args(0, '{"run_in_background": true}')
        assert result == {"run_in_background": False}


class TestFlushTaskArgBuffers:
    """Tests for ContentBlockManager.flush_task_arg_buffers."""

    def test_empty_buffer_skipped(self):
        cbm = ContentBlockManager()
        cbm.tool_states[0] = ToolCallState(
            block_index=0, tool_id="t1", name="Task", started=True
        )
        # No buffer accumulated yet.
        assert cbm.flush_task_arg_buffers() == []

    def test_already_emitted_skipped(self):
        cbm = ContentBlockManager()
        cbm.tool_states[0] = ToolCallState(
            block_index=0,
            tool_id="t1",
            name="Task",
            started=True,
            task_arg_buffer='{"a":1}',
            task_args_emitted=True,
        )
        assert cbm.flush_task_arg_buffers() == []

    def test_invalid_json_emits_empty_object(self):
        """JSONDecodeError path: invalid buffer -> '{}' fallback, state cleared."""
        cbm = ContentBlockManager()
        cbm.tool_states[0] = ToolCallState(
            block_index=0,
            tool_id="t1",
            name="Task",
            started=True,
            task_arg_buffer="not-json-at-all",
        )
        results = cbm.flush_task_arg_buffers()
        assert results == [(0, "{}")]
        assert cbm.tool_states[0].task_args_emitted is True
        assert cbm.tool_states[0].task_arg_buffer == ""

    def test_valid_json_serialised_with_run_in_background_false(self):
        cbm = ContentBlockManager()
        cbm.tool_states[0] = ToolCallState(
            block_index=0,
            tool_id="t1",
            name="Task",
            started=True,
            task_arg_buffer='{"description": "x"}',
        )
        results = cbm.flush_task_arg_buffers()
        assert len(results) == 1
        idx, payload = results[0]
        assert idx == 0
        parsed = json.loads(payload)
        assert parsed == {"description": "x", "run_in_background": False}

    def test_multiple_tools_independent(self):
        cbm = ContentBlockManager()
        cbm.tool_states[0] = ToolCallState(
            block_index=0,
            tool_id="t1",
            name="Task",
            started=True,
            task_arg_buffer='{"a":1}',
        )
        cbm.tool_states[1] = ToolCallState(
            block_index=1,
            tool_id="t2",
            name="Task",
            started=True,
            task_arg_buffer="garbage",
        )
        results = dict(cbm.flush_task_arg_buffers())
        assert json.loads(results[0]) == {"a": 1, "run_in_background": False}
        assert results[1] == "{}"
