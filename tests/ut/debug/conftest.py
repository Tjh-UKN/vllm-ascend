# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the device-independent guard without importing the NPU plugin."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from typing import Any, cast

import pytest

ROOT = Path(__file__).parents[3]


@pytest.fixture
def subject(monkeypatch):
    for name in ("vllm_ascend", "vllm_ascend.debug", "vllm"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    logger = ModuleType("vllm.logger")
    cast(Any, logger).logger = logging.getLogger("kv-guard-test")
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    for name, path in (
        ("vllm_ascend.config_utils", "config_utils.py"),
        ("vllm_ascend.debug.kv_guard_config", "debug/kv_guard_config.py"),
        ("vllm_ascend.debug.kv_guard", "debug/kv_guard.py"),
        ("vllm_ascend.debug.kv_guard_runtime", "debug/kv_guard_runtime.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, ROOT / "vllm_ascend" / path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def config():
    return NS(
        additional_config={"kv_guard_config": {"enabled": True}},
        model_config=NS(is_encoder_decoder=False),
        parallel_config=NS(prefill_context_parallel_size=1, decode_context_parallel_size=1, enable_dbo=False),
        compilation_config=NS(static_forward_context={}),
        speculative_config=None,
        kv_transfer_config=None,
    )


@pytest.fixture
def runtime(subject, config):
    guard = subject.create_guard(config)
    guard.events = []
    report = guard.report

    def recorded(event, **fields):
        guard.events.append({"event": event, **fields})
        report(event, **fields)

    guard.report = recorded
    guard.state.emit = recorded
    guard.state.configure({0: 4})
    guard.layer_groups = {"layer0": 0}
    return guard
