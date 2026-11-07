# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-runner ownership, explicit unsupported modes, and capture isolation."""

from types import SimpleNamespace as NS

import pytest

from .helpers import metadata, reasons


def runner_for(config, impl_name="AscendAttentionBackendImpl"):
    impl = type(impl_name, (), {})()
    config.compilation_config.static_forward_context = {"layer0": NS(impl=impl)}
    spec = NS(block_size=4, head_size=16)
    return NS(kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=spec, layer_names=["layer0"])])), impl


def test_disabled_has_no_runtime_or_config_side_effect(subject, config):
    config.additional_config = {}
    assert subject.create_guard(config) is None
    assert not hasattr(config, "_ascend_kv_guard")


def test_same_config_creates_independent_per_runner_state(subject, config):
    first, second = subject.create_guard(config), subject.create_guard(config)
    first.state.requests["r"] = ([2],)
    assert first is not second
    assert second.state.requests == {}
    assert not hasattr(config, "_ascend_kv_guard")


@pytest.mark.parametrize(
    "options", [{"enabled": "yes"}, {"enabled": 1}, {"action": "ignore"}, {"enabled": True, "typo": 1}]
)
def test_config_rejects_invalid_options(subject, config, options):
    config.additional_config = {"kv_guard_config": options}
    with pytest.raises(ValueError):
        subject.create_guard(config)


@pytest.mark.parametrize("impl_name", ["AscendAttentionBackendImpl", "AscendMLAImpl"])
def test_attention_only_validated_and_declared_model_state_bound(runtime, config, impl_name):
    runner, impl = runner_for(config, impl_name)
    runner.model_state = NS(kv_guard=None)
    runner.block_tables = NS(block_sizes=[4], kernel_block_sizes=[4])
    runtime.configure(runner)
    assert not hasattr(impl, "kv_guard")
    assert runner.model_state.kv_guard is runtime
    assert runtime.layer_groups == {"layer0": 0}


def test_empty_cache_groups_cannot_claim_guard_enabled(runtime):
    with pytest.raises(ValueError, match="no supported attention layers"):
        runtime.configure(NS(kv_cache_config=NS(kv_cache_groups=[])))


@pytest.mark.parametrize(
    "mode", ["connector", "cp", "dbo", "speculation", "encoder", "sliding", "compressed", "coordinates", "model_state"]
)
def test_unsupported_modes_fail_before_execution(runtime, config, mode):
    runner, _ = runner_for(config)
    spec = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec
    if mode == "connector":
        config.kv_transfer_config = NS(kv_connector="MooncakeConnector")
    elif mode == "cp":
        config.parallel_config.decode_context_parallel_size = 2
    elif mode == "dbo":
        config.parallel_config.enable_dbo = True
    elif mode == "speculation":
        config.speculative_config = NS()
    elif mode == "encoder":
        config.model_config.is_encoder_decoder = True
    elif mode == "sliding":
        spec.sliding_window = 16
    elif mode == "compressed":
        spec.compress_ratio = 4
    elif mode == "coordinates":
        runner.block_tables = NS(block_sizes=[4], kernel_block_sizes=[2])
    else:
        runner.model_state = NS()
    with pytest.raises(ValueError, match="unsupported"):
        runtime.configure(runner)


@pytest.mark.parametrize("full_graph", [False, True])
def test_dummy_live_write_uses_current_group_plan(runtime, full_graph):
    runtime.state.requests = {"r": ([2],)}
    with runtime.dummy_scope(NS()):
        runtime.observe_metadata({"layer0": metadata([9], [[2]], [1])}, [], full_graph)
    assert "dummy_write_to_live_block" in reasons(runtime)
    assert runtime.state.requests == {"r": ([2],)}


def test_dummy_scratch_is_allowed(runtime):
    runtime.state.requests = {"r": ([2],)}
    with runtime.dummy_scope(NS()):
        runtime.observe_metadata({"layer0": metadata([-1, 0, 12], [[0]], [0])}, [])
    assert not runtime.events


@pytest.mark.parametrize("phase", ["warmup", "capture"])
def test_startup_capture_never_materializes_tensor_metadata(runtime, monkeypatch, phase):
    monkeypatch.setattr(runtime, "_list", lambda _: pytest.fail("CPU metadata read during capture"))
    with runtime.dummy_scope(NS(), phase=phase):
        runtime.observe_metadata({"layer0": metadata([8], [[2]], [1])}, [], True)
    assert not runtime.events


def test_live_recapture_is_explicitly_rejected(runtime):
    runtime.state.requests = {"r": ([2],)}
    with pytest.raises(ValueError, match="no live requests"):
        runtime.check_capture()
