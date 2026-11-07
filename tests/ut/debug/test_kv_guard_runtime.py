# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One current execution plan per group is compared with an independent mirror."""

from types import SimpleNamespace as NS

import pytest
import torch

from .helpers import message, metadata, observe, reasons


@pytest.mark.parametrize("full_graph", [False, True])
def test_v2_prefix_append_is_a_metadata_plan_for_eager_and_full(runtime, full_graph):
    with runtime.execution(NS(), message(new=[("r", ([2, 3],), 3)], counts={"r": 3})):
        observe(runtime, ["r"], [11, 12, 13], [[2, 3]], [6], full_graph=full_graph)
    assert not runtime.events
    assert not hasattr(runtime, "write")
    assert not hasattr(runtime, "read")


@pytest.mark.parametrize("lengths,valid", [([1, 0, 0], True), ([1, 1, 0], False), ([], False)])
def test_full_graph_padding_requires_zero_lengths_after_real_rows(runtime, lengths, valid):
    with runtime.execution(NS(), message(new=[("r", ([2],), 0)], counts={"r": 1})):
        runtime.observe_metadata({"layer0": metadata([8], [[2], [0], [0]], lengths)}, ["r"], True)
    assert ("read_length_mismatch" not in reasons(runtime)) is valid


def test_corrupted_runner_table_cannot_change_shadow_plan(runtime):
    runner = NS(block_tables=NS(block_table=[[99]]))
    with runtime.execution(runner, message(new=[("r", ([2],), 0)], counts={"r": 1})):
        observe(runtime, ["r"], [396], [[99]], [1])
    assert "write_target_mismatch" in reasons(runtime)
    assert "read_block_mapping_mismatch" in reasons(runtime)
    assert runtime.state.requests["r"] == ([2],)


@pytest.mark.parametrize("full_graph", [False, True])
def test_swapped_requests_and_wrong_offsets_are_detected(runtime, full_graph):
    output = message(new=[("a", ([2],), 0), ("b", ([3],), 0)], counts={"a": 1, "b": 1})
    with runtime.execution(NS(), output):
        observe(runtime, ["b", "a"], [8, 12], [[3], [2]], [1, 1], full_graph=full_graph)
    assert "write_target_mismatch" in reasons(runtime)


@pytest.mark.parametrize(
    "slots,tables,lengths,reason",
    [
        ([10], [[2]], [2], "write_target_mismatch"),
        ([9], [[3]], [2], "read_block_mapping_mismatch"),
        ([9], [[2]], [1], "read_length_mismatch"),
        ([9], [[2]], [3], "read_length_mismatch"),
        ([], [[2]], [2], "write_tokens_not_observed"),
    ],
)
def test_exact_slot_read_table_and_read_length(runtime, slots, tables, lengths, reason):
    with runtime.execution(NS(), message(new=[("r", ([2],), 1)], counts={"r": 1})):
        observe(runtime, ["r"], slots, tables, lengths)
    assert reason in reasons(runtime)


def test_same_group_uses_only_first_known_non_none_metadata(runtime):
    runtime.layer_groups = {"layer0": 0, "layer1": 0, "layer2": 0}
    with runtime.execution(NS(), message(new=[("r", ([2],), 0)], counts={"r": 1})):
        runtime.observe_metadata(
            {"unknown": object(), "layer0": None, "layer1": metadata([8], [[2]], [1]), "layer2": object()},
            ["r"],
        )
    assert not runtime.events


def test_first_selected_invalid_metadata_does_not_fall_back_to_later_good_layer(runtime):
    runtime.layer_groups = {"layer0": 0, "layer1": 0}
    with runtime.execution(NS(), message(new=[("r", ([2],), 0)], counts={"r": 1})):
        runtime.observe_metadata({"layer0": NS(), "layer1": metadata([8], [[2]], [1])}, ["r"])
    assert "metadata_write_mapping_missing" in reasons(runtime)
    assert "metadata_read_range_missing" in reasons(runtime)


def test_other_layers_of_selected_group_do_not_require_metadata(runtime):
    runtime.layer_groups = {"layer0": 0, "layer1": 0}
    with runtime.execution(NS(), message(new=[("r", ([2],), 0)], counts={"r": 1})):
        observe(runtime, ["r"], [8], [[2]], [1])
    assert not runtime.events


def test_each_group_gets_its_own_expected_block_coordinates(runtime):
    runtime.state.configure({0: 4, 1: 8})
    runtime.layer_groups = {"layer0": 0, "layer1": 1}
    output = message(new=[("r", ([2], [5]), 1)], counts={"r": 1})
    with runtime.execution(NS(), output):
        runtime.observe_metadata({"layer0": metadata([9], [[2]], [2]), "layer1": metadata([41], [[5]], [2])}, ["r"])
    assert not runtime.events


def test_missing_group_reports_only_that_group(runtime):
    runtime.state.configure({0: 4, 1: 8})
    runtime.layer_groups = {"layer0": 0, "layer1": 1}
    with runtime.execution(NS(), message(new=[("r", ([2], [5]), 0)], counts={"r": 1})):
        observe(runtime, ["r"], [8], [[2]], [1])
    assert runtime.events == [{"event": "guard.coverage_gap", "reason": "metadata_group_missing", "group_id": 1}]


def test_metadata_is_read_once_per_group_only_after_final_parameters_change(runtime, monkeypatch):
    runtime.layer_groups = {"layer0": 0, "layer1": 0}
    current = metadata([99], [[2]], [1])
    reads = []
    original = runtime._list

    def recorded(value):
        reads.append(value)
        return original(value)

    monkeypatch.setattr(runtime, "_list", recorded)
    with runtime.execution(NS(), message(new=[("r", ([2],), 0)], counts={"r": 1})):
        runtime.observe_metadata({"layer0": current, "layer1": current}, ["r"], True)
        assert reads == []
        current.slot_mapping.copy_(torch.tensor([8]))
    assert sum(value.data_ptr() == current.slot_mapping.data_ptr() for value in reads) == 1
    assert not runtime.events


def test_only_actual_slot_prefix_is_materialized_from_padded_buffer(runtime, monkeypatch):
    current = metadata([8, 999, 999, 999], [[2]], [1])
    current.num_actual_tokens = 1
    materialized = []
    original = runtime._list

    def recorded(value):
        if value.data_ptr() == current.slot_mapping.data_ptr():
            materialized.append(value.numel())
        return original(value)

    monkeypatch.setattr(runtime, "_list", recorded)
    with runtime.execution(NS(), message(new=[("r", ([2],), 0)], counts={"r": 1})):
        runtime.observe_metadata({"layer0": current}, ["r"], True)
    assert materialized == [1]
    assert not runtime.events


def test_full_graph_requires_fresh_metadata_every_step(runtime):
    with runtime.execution(NS(), message(new=[("r", ([2],), 0)], counts={"r": 1})):
        observe(runtime, ["r"], [8], [[2]], [1], full_graph=True)
    with runtime.execution(NS(), message(cached=[("r", None, 1)], counts={"r": 1})):
        pass
    assert "metadata_group_missing" in reasons(runtime)


@pytest.mark.parametrize("batch", [[], ["r", "r"], ["other"]])
def test_batch_identity_mismatch_is_a_coverage_gap(runtime, batch):
    with runtime.execution(NS(), message(new=[("r", ([2],), 0)], counts={"r": 1})):
        observe(runtime, batch, [8], [[2]], [1])
    assert "batch_request_mismatch" in reasons(runtime)


def test_actual_batch_ids_must_be_explicitly_observed(runtime):
    runner = NS(input_batch=NS(req_ids=["r"]))
    with runtime.execution(runner, message(new=[("r", ([2],), 0)], counts={"r": 1})):
        runtime.observe_metadata({"layer0": metadata([8], [[2]], [1])})
    assert "batch_request_ids_missing" in reasons(runtime)


def test_no_token_step_can_release_without_a_forward(runtime):
    runtime.state.requests = {"r": ([2],)}
    with runtime.execution(NS(), message(finished=["r"])):
        pass
    assert runtime.state.requests == {}
    assert not runtime.events


def test_no_token_step_does_not_accept_an_active_metadata_plan(runtime):
    with runtime.execution(NS(), message()):
        observe(runtime, [], [8], [[2]], [1])
    assert "unscheduled_attention_plan" in reasons(runtime)


def test_native_exception_preserved_without_audit_and_metadata_references_cleared(runtime):
    with (
        pytest.raises(RuntimeError, match="native failure"),
        runtime.execution(NS(), message(new=[("r", ([2],), 0)], counts={"r": 1})),
    ):
        observe(runtime, ["r"], [99], [[2]], [1])
        raise RuntimeError("native failure")
    assert not runtime.events
    assert runtime._phase is None
    assert not runtime._metadata


def test_raise_is_synchronous_and_sticky_without_modifying_output_getter(runtime, subject):
    runtime.action = "raise"
    output = NS(get_output=lambda: "result")
    original = output.get_output
    with (
        pytest.raises(subject.KVGuardError, match="write_target_mismatch"),
        runtime.execution(NS(), message(new=[("r", ([2],), 0)], counts={"r": 1})),
    ):
        observe(runtime, ["r"], [9], [[2]], [1], full_graph=True)
    assert output.get_output is original
    with pytest.raises(subject.KVGuardError, match="restart the worker"):
        runtime.check_failure()


def test_log_reports_group_execution_plan_evidence(runtime, caplog):
    with runtime.execution(NS(), message(new=[("r", ([2],), 0)], counts={"r": 1})):
        observe(runtime, ["r"], [9], [[2]], [1])
        result = "native result"
    runtime.check_failure()
    assert result == "native result"
    assert "group_id=0" in caplog.text
    assert "evidence=execution_plan" in caplog.text
    assert "layer=" not in caplog.text
