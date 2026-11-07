# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare one attention execution plan per cache group with SchedulerOutput.

Eager and FULL replay use this step's final metadata. The checker does not
observe operator arguments, kernel memory accesses, KV contents or generations.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

from vllm_ascend.debug.kv_guard import KVGuard
from vllm_ascend.debug.kv_guard_config import KVGuardConfig

_MAX_REPORT_KEYS = 1024


class KVGuardError(RuntimeError):
    """A violation or missing coverage prevents further model output."""


class GuardRuntime:
    def __init__(self, vllm_config: Any):
        from vllm.logger import logger

        self.logger = logger
        self.config = vllm_config
        self.action = KVGuardConfig.from_vllm_config(vllm_config).action
        self.state = KVGuard(self.report)
        self.layer_groups: dict[str, int] = {}
        self._failure: str | None = None
        self._reported: OrderedDict[tuple, None] = OrderedDict()
        self._phase: str | None = None
        self._reset_step()

    def _reset_step(self) -> None:
        self._metadata: dict[int, Any] = {}
        self._request_ids: list[str] | None = None

    def report(self, event: str, **fields: Any) -> None:
        failed = event in ("guard.violation", "guard.coverage_gap")
        if failed:
            fields.setdefault("evidence", "execution_plan")
            if self._failure is None:
                self._failure = str(fields.get("reason", event))
            key = event, tuple((name, repr(value)) for name, value in sorted(fields.items()))
            if key in self._reported:
                self._reported.move_to_end(key)
                return
            self._reported[key] = None
            if len(self._reported) > _MAX_REPORT_KEYS:
                self._reported.popitem(last=False)
        payload = " ".join(f"{key}={value}" for key, value in fields.items())
        log = self.logger.error if failed else self.logger.info
        log("[KV_GUARD] %s %s", event, payload)

    def gap(self, reason: str, **fields: Any) -> None:
        self.report("guard.coverage_gap", reason=reason, **fields)

    def check_failure(self) -> None:
        if self.action == "raise" and self._failure is not None:
            raise KVGuardError(f"KV guard stopped model output: {self._failure}; restart the worker before reuse")

    def configure(self, runner: Any) -> None:
        config, parallel = self.config, self.config.parallel_config
        if any(
            (
                config.kv_transfer_config is not None,
                getattr(config.model_config, "is_encoder_decoder", False),
                config.speculative_config is not None,
                getattr(parallel, "prefill_context_parallel_size", 1) != 1,
                getattr(parallel, "decode_context_parallel_size", 1) != 1,
                getattr(parallel, "enable_dbo", False),
            )
        ):
            raise ValueError("KV guard: unsupported connector, CP, DBO, speculation or encoder/decoder mode")
        sizes = {}
        self.layer_groups = {}
        for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            sizes[group_id] = group.kv_cache_spec.block_size
            for name in group.layer_names:
                spec = getattr(group.kv_cache_spec, "kv_cache_specs", {}).get(name, group.kv_cache_spec)
                if (
                    not hasattr(spec, "head_size")
                    or getattr(spec, "sliding_window", None) is not None
                    or getattr(spec, "attention_chunk_size", None) is not None
                    or getattr(spec, "compress_ratio", 1) != 1
                    or type(spec).__name__ == "CircularBufferSpec"
                ):
                    raise ValueError(f"KV guard: unsupported cache specification: {type(spec).__name__}")
                layer = config.compilation_config.static_forward_context.get(name)
                impl = getattr(layer, "impl", None)
                if type(impl).__name__ not in ("AscendAttentionBackendImpl", "AscendMLAImpl"):
                    raise ValueError(f"KV guard: unsupported attention implementation: {type(impl).__name__}")
                self.layer_groups[name] = group_id
        if not self.layer_groups:
            raise ValueError("KV guard: no supported attention layers")
        tables = getattr(runner, "block_tables", None)
        if tables is not None and (
            list(tables.block_sizes) != list(sizes.values()) or list(tables.kernel_block_sizes) != list(sizes.values())
        ):
            raise ValueError("KV guard: unsupported allocator/kernel block coordinates")
        model_state = getattr(runner, "model_state", None)
        if model_state is None or not hasattr(model_state, "kv_guard"):
            raise ValueError("KV guard: unsupported V2 model state")
        self.state.configure(sizes)
        model_state.kv_guard = self
        self.report("guard.enabled", failure_action=self.action)

    def observe_metadata(self, metadata: Any, request_ids: list[str] | None = None, full_graph: bool = False) -> None:
        if self._phase not in ("forward", "dummy"):
            return
        self._metadata = {}
        self._request_ids = list(request_ids) if request_ids is not None else None
        if isinstance(metadata, dict):
            for name, item in metadata.items():
                group = self.layer_groups.get(name)
                if group is not None and item is not None:
                    self._metadata.setdefault(group, item)

    @staticmethod
    def _list(value: Any) -> list:
        if isinstance(value, (list, tuple)):
            return list(value)
        return value.detach().cpu().tolist()

    @staticmethod
    def _field(metadata: Any, *names: str) -> Any:
        return next((value for name in names if (value := getattr(metadata, name, None)) is not None), None)

    def _slots(self, metadata: Any, group: int) -> list[int] | None:
        value = self._field(metadata, "slot_mapping")
        if value is None:
            self.gap("metadata_write_mapping_missing", group_id=group)
            return None
        count = getattr(metadata, "num_actual_tokens", None)
        slots = self._list(value if count is None else value[:count])
        slots = [slot for row in slots for slot in (row if isinstance(row, list) else [row])]
        return slots

    def _audit(self) -> None:
        if not self.state.scheduled:
            for group, metadata in self._metadata.items():
                slots = self._slots(metadata, group)
                lengths = self._field(metadata, "seq_lens_gpu", "seq_lens", "seq_lens_list", "seq_lens_cpu")
                if (slots is not None and any(slot >= 0 for slot in slots)) or (
                    lengths is not None and any(self._list(lengths))
                ):
                    self.report("guard.violation", reason="unscheduled_attention_plan", group_id=group)
            return
        request_ids = self._request_ids
        for group in self.state.group_sizes:
            metadata = self._metadata.get(group)
            if metadata is None:
                self.gap("metadata_group_missing", group_id=group)
                continue
            if request_ids is None:
                self.gap("batch_request_ids_missing", group_id=group)
                continue
            if len(request_ids) != len(set(request_ids)) or set(request_ids) != set(self.state.scheduled):
                self.gap("batch_request_mismatch", group_id=group)
                continue
            plan = self.state.expected(group, request_ids)
            if plan is None:
                continue
            slots = self._slots(metadata, group)
            if slots is not None:
                if slots[: len(plan.slots)] != plan.slots or any(slot >= 0 for slot in slots[len(plan.slots) :]):
                    self.report("guard.violation", reason="write_target_mismatch", group_id=group)
                if len(slots) < len(plan.slots):
                    self.gap("write_tokens_not_observed", group_id=group)
            table = self._field(metadata, "block_tables", "block_table", "block_table_tensor")
            lengths = self._field(metadata, "seq_lens_gpu", "seq_lens", "seq_lens_list", "seq_lens_cpu")
            if table is None or lengths is None:
                self.gap("metadata_read_range_missing", group_id=group)
                continue
            lengths = self._list(lengths)
            rows = len(plan.lengths)
            if lengths[:rows] != plan.lengths or any(lengths[rows:]):
                self.report("guard.violation", reason="read_length_mismatch", group_id=group)
            columns = max((len(row) for row in plan.tables), default=0)
            table = (
                [list(row[:columns]) for row in table[:rows]]
                if isinstance(table, (list, tuple))
                else self._list(table[:rows, :columns])
            )
            if len(table) < rows or any(
                actual[: len(expected)] != expected for actual, expected in zip(table, plan.tables)
            ):
                self.report("guard.violation", reason="read_block_mapping_mismatch", group_id=group)

    @contextmanager
    def execution(self, runner: Any, scheduler_output: Any):
        self.check_failure()
        self.state.bind(scheduler_output)
        self._reset_step()
        self._phase = "forward"
        try:
            self.check_failure()
            yield
            try:
                self._audit()
            except Exception as exc:
                self.gap(f"access_audit_failed:{type(exc).__name__}")
            self.check_failure()
        finally:
            self._phase = None
            self._reset_step()

    def check_capture(self) -> None:
        self.check_failure()
        if self.state.requests:
            raise ValueError("KV guard: capture requires no live requests")

    @contextmanager
    def dummy_scope(self, runner: Any, phase: str = "dummy"):
        self.check_failure()
        if phase == "capture":
            self.check_capture()
        previous = self._phase, self._metadata, self._request_ids
        self._reset_step()
        self._phase = phase
        try:
            yield
            if phase == "dummy":
                for group, size in self.state.group_sizes.items():
                    metadata = self._metadata.get(group)
                    if metadata is None:
                        if self.state.requests:
                            self.gap("dummy_write_plan_missing", group_id=group)
                        continue
                    slots = self._slots(metadata, group)
                    live = self.state.live_blocks(group)
                    if slots is not None and any(slot >= 0 and slot // size in live for slot in slots):
                        self.report("guard.violation", reason="dummy_write_to_live_block", group_id=group)
                self.check_failure()
        finally:
            self._phase, self._metadata, self._request_ids = previous


def create_guard(vllm_config: Any) -> GuardRuntime | None:
    if not KVGuardConfig.from_vllm_config(vllm_config).enabled:
        return None
    return GuardRuntime(vllm_config)
