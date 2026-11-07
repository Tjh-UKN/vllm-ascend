# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shadow bindings use standard scheduler messages, never runner tables."""

import pytest

from .helpers import message, reasons


def test_new_plan_copies_scheduler_blocks_and_accepts_framework_prefix(runtime):
    blocks = ([2, 3],)
    runtime.state.bind(message(new=[("r", blocks, 3)], counts={"r": 3}))
    blocks[0][0] = 99
    plan = runtime.state.expected(0, ["r"])
    assert plan.slots == [11, 12, 13]
    assert plan.tables == [[2, 3]]
    assert plan.lengths == [6]
    assert not runtime.events


def test_append_none_resume_preempt_finish_and_request_id_reuse(runtime):
    state = runtime.state
    state.bind(message(new=[("r", ([2],), 0)], counts={"r": 3}))
    state.bind(message(cached=[("r", ([3],), 3)], counts={"r": 3}))
    assert state.expected(0, ["r"]).slots == [11, 12, 13]
    state.bind(message(cached=[("r", None, 6)], counts={"r": 1}))
    assert state.expected(0, ["r"]).slots == [14]
    state.bind(message(preempted=["r"]))
    assert state.requests == {}
    state.bind(message(cached=[("r", ([7],), 0)], resumed=["r"], counts={"r": 2}))
    assert state.expected(0, ["r"]).slots == [28, 29]
    state.bind(message(finished=["r"]))
    assert state.requests == {}
    state.bind(message(new=[("r", ([11],), 0)], counts={"r": 1}))
    assert state.expected(0, ["r"]).slots == [44]


def test_multiple_groups_and_actual_batch_reordering(runtime):
    state = runtime.state
    state.configure({0: 4, 1: 8})
    state.bind(message(new=[("a", ([2], [5]), 1), ("b", ([3], [7]), 0)], counts={"a": 2, "b": 1}))
    assert state.expected(0, ["b", "a"]).slots == [12, 9, 10]
    assert state.expected(1, ["b", "a"]).slots == [56, 41, 42]


@pytest.mark.parametrize("blocks,computed,count", [([], 0, 1), ([0], 0, 1), ([2], -1, 1), ([2], 0, 0)])
def test_incomplete_or_invalid_scheduler_plan_is_a_gap(runtime, blocks, computed, count):
    runtime.state.bind(message(new=[("r", (blocks,), computed)], counts={"r": count}))
    assert runtime.state.expected(0, ["r"]) is None
    assert any(event["event"] == "guard.coverage_gap" for event in runtime.events)


def test_missing_cached_or_resumed_binding_is_not_reconstructed_from_runner(runtime):
    runtime.state.bind(message(cached=[("r", ([2],), 0)], counts={"r": 1}))
    assert runtime.state.expected(0, ["r"]) is None
    assert "cached_request_binding_missing" in reasons(runtime)
    runtime.state.bind(message(cached=[("r", None, 0)], resumed=["r"], counts={"r": 1}))
    assert "resume_block_mapping_missing" in reasons(runtime)
