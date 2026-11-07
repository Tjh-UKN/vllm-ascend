# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent request/block mirror built only from standard SchedulerOutput.

Computed prefixes are framework inputs, not proof of KV contents or allocator
generations. This mirror checks execution plans, not the scheduler's allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AccessPlan:
    slots: list[int]
    tables: list[list[int]]
    lengths: list[int]


class KVGuard:
    def __init__(self, emit: Any):
        self.emit = emit
        self.requests: dict[str, tuple[list[int], ...]] = {}
        self.scheduled: dict[str, tuple[int, int]] = {}
        self.group_sizes: dict[int, int] = {}

    def configure(self, group_sizes: dict[int, int]) -> None:
        self.group_sizes = dict(group_sizes)

    @staticmethod
    def _copy_groups(groups: Any) -> tuple[list[int], ...]:
        return tuple(list(group) for group in groups)

    def bind(self, output: Any) -> None:
        """Apply the scheduler's append/replace/release contract before execution."""
        for request_id in output.finished_req_ids | (output.preempted_req_ids or set()):
            self.requests.pop(request_id, None)
        computed = {}
        for request in output.scheduled_new_reqs:
            self.requests[request.req_id] = self._copy_groups(request.block_ids)
            computed[request.req_id] = request.num_computed_tokens
        cached = output.scheduled_cached_reqs
        for index, request_id in enumerate(cached.req_ids):
            groups = cached.new_block_ids[index]
            if request_id in cached.resumed_req_ids:
                if groups is None:
                    self.emit("guard.coverage_gap", reason="resume_block_mapping_missing", request_id=request_id)
                    self.requests.pop(request_id, None)
                else:
                    self.requests[request_id] = self._copy_groups(groups)
            elif request_id not in self.requests:
                self.emit("guard.coverage_gap", reason="cached_request_binding_missing", request_id=request_id)
            elif groups is not None:
                current = self.requests[request_id]
                if len(current) != len(groups):
                    self.emit("guard.coverage_gap", reason="cache_group_count_mismatch", request_id=request_id)
                    self.requests.pop(request_id, None)
                else:
                    self.requests[request_id] = tuple(old + list(new) for old, new in zip(current, groups))
            computed[request_id] = cached.num_computed_tokens[index]
        self.scheduled = {
            request_id: (computed.get(request_id, -1), count)
            for request_id, count in output.num_scheduled_tokens.items()
        }

    def expected(self, group_id: int, request_ids: list[str]) -> AccessPlan | None:
        slots: list[int] = []
        tables: list[list[int]] = []
        lengths: list[int] = []
        size = self.group_sizes[group_id]
        for request_id in request_ids:
            computed, count = self.scheduled.get(request_id, (-1, 0))
            groups = self.requests.get(request_id, ())
            needed = (computed + count + size - 1) // size
            if computed < 0 or count <= 0 or group_id >= len(groups) or len(groups[group_id]) < needed:
                self.emit(
                    "guard.coverage_gap", reason="scheduler_plan_incomplete", request_id=request_id, group_id=group_id
                )
                return None
            blocks = groups[group_id][:needed]
            if any(block <= 0 for block in blocks):
                self.emit(
                    "guard.coverage_gap", reason="scheduler_block_id_invalid", request_id=request_id, group_id=group_id
                )
                return None
            tables.append(list(blocks))
            lengths.append(computed + count)
            slots.extend(blocks[token // size] * size + token % size for token in range(computed, computed + count))
        return AccessPlan(slots, tables, lengths)

    def live_blocks(self, group_id: int) -> set[int]:
        return {block for groups in self.requests.values() if group_id < len(groups) for block in groups[group_id]}
