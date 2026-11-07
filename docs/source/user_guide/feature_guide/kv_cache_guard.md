# KV cache address guard

KV Guard in Model Runner V2 compares the scheduler's declared token destinations with the runner's
final attention metadata, once per KV cache group. Eager and FULL graph modes
use the same metadata-plan checks. Attention implementations are unchanged;
operator arguments and per-layer execution order are not recorded.

## Enable the guard

The guard is disabled by default and requires Model Runner V2. Enable it for an engine with:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve MODEL --additional-config '{"kv_guard_config":{"enabled":true,"action":"raise"}}'
```

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable worker metadata-plan checks. |
| `action` | `"log"` or `"raise"` | `"log"` | Report failures or stop returning worker results. |

The Python API accepts the same `additional_config` dictionary. Unknown fields,
non-boolean `enabled` values and unknown actions are rejected. Each runner owns
its guard; the configuration object does not hold mutable checking state.
Enabling the guard with Model Runner V1 raises a configuration error.

`raise` latches the first violation or incomplete audit. Catching an exception
does not reset the guard; restart the worker before reuse. There is no collective
failure coordination across workers.

## Independent execution plan

The worker consumes the unmodified `SchedulerOutput` before its runner updates:

- New requests supply complete block tables and computed-token counts.
- Cached requests supply block-table increments, or replacements for resumed
  requests, and current computed-token counts.
- Scheduled token counts define the token positions authorized for this step.
- Finished and preempted request IDs retire the corresponding bindings.

The guard keeps its own request-to-block mirror. It derives expected slots,
read block tables and sequence lengths from these scheduler messages and the
actual batch order. It does not use the runner's address metadata to construct
its expected plan.

For each KV cache group, the guard selects the first non-null metadata item
whose layer belongs to that group, matching the current group-level audit.
Other layers in that group are not independently audited. A missing group,
incomplete plan or incorrect batch binding reports a coverage gap.

Checks compare exact token slots, logical read block order and declared read
lengths. For example, swapping A's and B's slots is an error even if both target
blocks exist. Negative slot padding and zero-length padded request rows are
allowed; padding must not authorize extra writes or reads.

FULL replay uses this execution's final metadata after graph parameter updates.
Captured or previous-step plans cannot substitute for the current plan. V2
uses the actual batch request order, not persistent request-state indices.

Runner and model-state implementations call the guard explicitly. No Scheduler
or allocator patch is installed, and runner, connector and asynchronous output
methods are not replaced at runtime.

## Coverage boundaries

Dense PA and MLA append-only cache paths in Runner V2 are adapted. The guard
checks consistency between scheduler messages and selected metadata. It does
not prove that a particular layer executed, that operator arguments matched
the metadata, or that a kernel used those addresses.

Allocator generations, initialized KV contents, historical prefix writes and
transport integrity are outside this revision's scope. A computed prefix is
a framework input, not proof that its bytes were initialized correctly. Prefix
reuse retains address-plan checks without validating the reused contents.

KV connectors, CP, DBO, speculative decoding, encoder/decoder execution,
sliding/circular caches, compressed/state caches and unadapted attention or
model-state implementations are rejected. There is no remote-load submission,
completion, cancellation or failed-load recovery guarantee.

Startup graph capture does not read device metadata on the CPU. Recapture with
live request state is rejected. Dummy writes can be checked against request
blocks currently known to the mirror; this is not an allocator-liveness proof.

The guard cannot prevent or undo writes already submitted to the device, verify
KV numerical values, or establish model-output correctness.

## Failure reporting and performance

Logs use `[KV_GUARD]`. Plan mismatches use `guard.violation` with
`evidence=execution_plan` and a cache group when available.
`guard.coverage_gap` means required evidence is missing, not a successful check.
Reports are deduplicated with bounded state.

Audits complete synchronously before guarded execution returns. Copying address
metadata to CPU can synchronize the NPU. No deferred audit is attached to an
asynchronous output. Performance results for earlier deferred implementations
do not apply; measure this revision on the intended workload.

## Validation

Run focused tests with:

```bash
pytest -q --confcutdir=tests/ut/debug tests/ut/debug
```

The service regression compares disabled/enabled greedy outputs for Runner V2,
eager/FULL, repeated prompts and concurrent requests:

```bash
pytest -q tests/e2e/pull_request/one_card/test_kv_cache_guard.py
```

CPU tests establish checker behavior, not NPU service correctness. Record
exact-head NPU service runs, current replay metadata and controlled plan faults
before merge. Standalone NPU metadata or synthetic graph checks do not replace
a full model-service run. Allocator-generation or transport guarantees require
an upstream vLLM lifecycle contract rather than Scheduler patches in Ascend.
