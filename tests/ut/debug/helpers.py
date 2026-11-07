# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import torch


def message(*, new=(), cached=(), resumed=(), counts=None, finished=(), preempted=()):
    return NS(
        scheduled_new_reqs=[
            NS(req_id=req, block_ids=groups, num_computed_tokens=computed) for req, groups, computed in new
        ],
        scheduled_cached_reqs=NS(
            req_ids=[req for req, _, _ in cached],
            new_block_ids=[groups for _, groups, _ in cached],
            num_computed_tokens=[computed for _, _, computed in cached],
            resumed_req_ids=set(resumed),
        ),
        num_scheduled_tokens=counts or {},
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
    )


def metadata(slots, tables, lengths):
    return NS(
        slot_mapping=torch.tensor(slots),
        block_tables=torch.tensor(tables),
        seq_lens=torch.tensor(lengths),
        num_actual_tokens=len(slots),
    )


def observe(guard, request_ids, slots, tables, lengths, *, full_graph=False):
    guard.observe_metadata({"layer0": metadata(slots, tables, lengths)}, request_ids, full_graph)


def reasons(guard):
    return [event.get("reason") for event in guard.events]
