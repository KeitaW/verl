# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Precision-agnostic breakdown of a rollout weight-resync (``update_weights``).

Why this module exists
----------------------
``timing_s/update_weights`` is a single wall-clock number for the whole
weight resync. On a large MoE model it is one of the biggest non-generation
line items in a step, but nothing in the tree reports *where* that time
goes for a plain BF16 run.

verl already grew a detailed 8-phase probe for the FP8 layerwise-reload path
(``verl/utils/vllm/fp8_phase_timer.py`` on the FP8 branch). That probe cannot
answer the BF16 question, by construction: the only place it is installed is
``begin_fp8_layerwise_reload()``, which
``vllm_rollout/utils.py:update_weights_from_ipc`` calls exclusively inside
``if is_fp8_model(self.model_runner.vllm_config)``. Every one of its phase
blocks is written as ``timer = get_current_phase_timer(); if timer is None:
<uninstrumented path>``, so on a BF16 sync the timer is always ``None`` and
every phase falls through untimed.

This module is the missing half: a *sender-side*, precision-agnostic
stopwatch that decomposes the resync into the segments the sender itself
controls, so a BF16 run produces a breakdown that sums back to the outer
``update_weights`` number with an explicit unaccounted remainder.

Design constraints
------------------
* **Off by default.** Enabled only by ``VERL_RESYNC_BREAKDOWN=1``. When
  disabled, every entry point is a single ``is_enabled()`` bool check and no
  object is allocated, so the un-instrumented stack is untouched.
* **No hot-path pollution when on.** The per-tensor loop does NOT get a
  timer call per tensor (that would add tens of thousands of clock reads to
  the very loop under study and change what is being measured). Instead the
  per-tensor cost is obtained by *difference*: the loop's total span minus
  the bucket-flush spans measured inside it. Counters (tensor count, bytes,
  bucket count) are plain integer increments.
* **Honest about what it cannot separate.** ``record_unaccounted()`` makes
  the residual an explicit reported field rather than silently distributing
  it across the named segments.

Segments
--------
``s1_socket_init_s``   sender ZMQ socket create+bind
``s2_buffer_init_s``   bucket buffer alloc + IPC handle export + receiver ack
``s3a_source_pull_s``  advancing the weight source ONE tensor at a time. In
                       production this is ``bridge.export_hf_weights(module)``
                       (``verl/workers/engine/megatron/transformer_impl.py:826``),
                       i.e. re-materialising each Megatron param in HF layout,
                       which for a TP/PP/EP-sharded actor means collectives per
                       tensor. This is the term the FP8 probe could not see and
                       the one most likely to dominate.
``s3b_bucket_fill_s``  local work per tensor: metadata dict insert + the
                       ``view().view().copy_()`` into the bucket buffer
``s4_bucket_flush_s``  sum of the mid-loop bucket flushes: device
                       synchronize + send_pyobj + wait for receiver ack
``s5_final_flush_s``   the last bucket's synchronize + send + ack
``s6_cleanup_s``       socket close + buffer release + gc.collect +
                       ipc_collect + empty_cache
``s7_direct_large_s``  over-sized weights sent by their own IPC handle instead
                       of through the bucket (``_direct_send_large_weight``)

Every segment is disjoint, so ``accounted_s`` is a plain sum and
``unaccounted_s = wall_total_s - accounted_s`` is a real residual rather than a
double-counting artifact.
"""

import json
import logging
import os
import time
from contextlib import contextmanager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

ENV_FLAG = "VERL_RESYNC_BREAKDOWN"

SEGMENT_KEYS = (
    "s1_socket_init_s",
    "s2_buffer_init_s",
    "s3a_source_pull_s",
    "s3b_bucket_fill_s",
    "s4_bucket_flush_s",
    "s5_final_flush_s",
    "s6_cleanup_s",
    "s7_direct_large_s",
)


def is_enabled() -> bool:
    """True when the breakdown is switched on via the environment.

    Read on every call rather than cached at import so a driver can flip it
    per-process without re-importing verl.
    """
    return os.getenv(ENV_FLAG, "0") not in ("0", "", "false", "False")


class ResyncBreakdown:
    """Accumulates one weight-resync's segment timings and counters.

    One instance per ``async_send_weights()`` call. Not thread-safe by
    design: a sender instance belongs to one worker's one sync.
    """

    __slots__ = (
        "segments",
        "n_tensors",
        "n_noncontiguous",
        "n_buckets",
        "n_direct_large",
        "total_bytes",
        "_wall_start",
        "wall_total_s",
        "bucket_size_mb",
        "rank",
    )

    def __init__(self, bucket_size_mb: int | None = None, rank: int | None = None):
        self.segments: dict[str, float] = {k: 0.0 for k in SEGMENT_KEYS}
        self.n_tensors = 0
        self.n_noncontiguous = 0
        self.n_buckets = 0
        self.n_direct_large = 0
        self.total_bytes = 0
        self._wall_start = None
        self.wall_total_s = 0.0
        self.bucket_size_mb = bucket_size_mb
        self.rank = rank

    # -- lifecycle ------------------------------------------------------
    def start(self):
        self._wall_start = time.perf_counter()

    def stop(self):
        if self._wall_start is not None:
            self.wall_total_s = time.perf_counter() - self._wall_start

    @contextmanager
    def segment(self, key: str):
        """Time a named segment, accumulating (segments may repeat)."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.segments[key] += time.perf_counter() - t0

    # Exposed so the hot loop can take raw timestamps without paying for a
    # context manager (generator create + two frame switches) per tensor.
    clock = staticmethod(time.perf_counter)

    def add(self, key: str, dt: float):
        """Accumulate a pre-measured duration into a segment."""
        self.segments[key] += dt

    # -- counters (cheap; safe inside the per-tensor loop) ---------------
    def count_tensor(self, nbytes: int, contiguous: bool = True):
        self.n_tensors += 1
        self.total_bytes += nbytes
        if not contiguous:
            self.n_noncontiguous += 1

    def count_bucket(self):
        self.n_buckets += 1

    def count_direct_large(self):
        self.n_direct_large += 1

    # -- reporting ------------------------------------------------------
    @property
    def per_tensor_loop_s(self) -> float:
        """Per-tensor work: source pull + bucket fill, excluding flushes.

        Both terms are now measured directly rather than recovered by
        subtracting nested spans, so this is a plain sum.
        """
        return self.segments["s3a_source_pull_s"] + self.segments["s3b_bucket_fill_s"]

    @property
    def accounted_s(self) -> float:
        """Sum of all segments. They are disjoint by construction."""
        return sum(self.segments.values())

    @property
    def unaccounted_s(self) -> float:
        """Sender wall time not covered by any named segment.

        Deliberately surfaced instead of being folded into a segment: the
        sender cannot see the receiver-side load, nor the caller's own
        pre/post work, so a non-trivial residual is expected and must be
        readable as such.
        """
        return self.wall_total_s - self.accounted_s

    def as_dict(self) -> dict:
        d = {
            "resync_breakdown": True,
            "rank": self.rank,
            "bucket_size_mb": self.bucket_size_mb,
            "wall_total_s": round(self.wall_total_s, 6),
            "n_tensors": self.n_tensors,
            "n_noncontiguous_tensors": self.n_noncontiguous,
            "n_buckets": self.n_buckets,
            "n_direct_large_sends": self.n_direct_large,
            "total_bytes": self.total_bytes,
        }
        for k in SEGMENT_KEYS:
            d[k] = round(self.segments[k], 6)
        d["per_tensor_loop_s"] = round(self.per_tensor_loop_s, 6)
        if self.n_tensors:
            d["per_tensor_mean_ms"] = round(self.per_tensor_loop_s / self.n_tensors * 1e3, 6)
            d["source_pull_mean_ms"] = round(self.segments["s3a_source_pull_s"] / self.n_tensors * 1e3, 6)
            d["bucket_fill_mean_ms"] = round(self.segments["s3b_bucket_fill_s"] / self.n_tensors * 1e3, 6)
        d["accounted_s"] = round(self.accounted_s, 6)
        d["unaccounted_s"] = round(self.unaccounted_s, 6)
        if self.wall_total_s > 0:
            d["unaccounted_pct"] = round(100.0 * self.unaccounted_s / self.wall_total_s, 3)
            # The headline number this probe exists to produce: what fraction of
            # the sender's wall time each stage owns.
            d["pct"] = {k: round(100.0 * self.segments[k] / self.wall_total_s, 3) for k in SEGMENT_KEYS}
        return d

    def emit(self):
        """Log one JSON line. One line per sender rank per sync."""
        try:
            logger.warning("RESYNC_BREAKDOWN %s", json.dumps(self.as_dict(), sort_keys=True))
        except Exception as exc:  # never let instrumentation break a run
            logger.warning("RESYNC_BREAKDOWN emit failed: %r", exc)


def maybe_new(bucket_size_mb: int | None = None, rank: int | None = None) -> "ResyncBreakdown | None":
    """Return a fresh accumulator when enabled, else ``None``.

    Call sites are then a uniform ``if bd is not None:`` / ``with
    nullcontext()`` pattern, so the disabled path costs one env lookup.
    """
    if not is_enabled():
        return None
    return ResyncBreakdown(bucket_size_mb=bucket_size_mb, rank=rank)


# ---------------------------------------------------------------------------
# Orchestration side
# ---------------------------------------------------------------------------
# The segments above are all *sender-local*: they describe what one actor rank
# does between "socket bound" and "buffer freed". That is NOT what the trainer's
# ``timing_s/update_weights`` measures.
#
# ``marked_timer("update_weights")`` (verl/trainer/ppo/ray_trainer.py:1690-1691)
# wraps ``CheckpointEngineManager.update_weights()``
# (verl/checkpoint_engine/base.py:486-538), which is an EIGHT STAGE
# orchestration: abort in-flight requests, build a temporary RayWorkerGroup,
# free the KV cache, build the transfer process group, run the actual weight
# send/receive under one ``ray.get`` over every rank, finalize, restore the KV
# cache, resume generation.
#
# Seven of those eight stages are not weight movement at all, and none of them
# is visible to a sender-side timer. Measuring only the sender therefore
# reproduces the exact failure this card was opened to fix: a breakdown whose
# residual is ~all of the wall time.
#
# Cost of this instrumentation: ~12 clock reads per sync in the single trainer
# driver process (NOT per tensor, NOT per rank), against a ~100 s span. It is
# unconditional-safe, but still gated on the same env flag so the default stack
# is bit-identical.

ORCH_KEYS = (
    "o1_abort_replicas_s",
    "o2_worker_group_s",
    "o3_release_kv_cache_s",
    "o4a_prepare_s",
    "o4b_build_topology_s",
    "o4c_init_process_group_s",
    "o5_send_recv_barrier_s",
    "o6_finalize_s",
    "o7_resume_kv_cache_s",
    "o8_resume_generation_s",
)


class OrchestrationBreakdown:
    """Times the 8 stages of ``CheckpointEngineManager.update_weights``.

    One instance per resync, living in the trainer driver process. This is the
    outer half of the breakdown: it is what makes the numbers sum back to
    ``timing_s/update_weights`` instead of to one rank's sender span.

    Interpreting ``o5_send_recv_barrier_s`` is the point of the whole probe.
    It is a single ``ray.get`` over actor ranks *and* rollout ranks
    (base.py:515-518), so it is a full-fleet barrier: it cannot finish before
    the slowest rank. Combined with the per-rank sender ``wall_total_s`` from
    ``ResyncBreakdown`` it splits three ways --

        o5 ~= max_rank(sender wall)          -> the transfer itself dominates
        o5 >> max_rank(sender wall)          -> straggler / barrier / receiver
                                                load dominates
        o5 spread across ranks is wide       -> imbalance, not raw bandwidth

    which is exactly the "collective 通信 / 同期待ち (barrier/straggler)"
    decomposition the card asks for, and which no sender-local timer can give.
    """

    __slots__ = ("segments", "_wall_start", "wall_total_s", "global_steps", "backend")

    def __init__(self, global_steps: int | None = None, backend: str | None = None):
        self.segments: dict[str, float] = {k: 0.0 for k in ORCH_KEYS}
        self._wall_start = None
        self.wall_total_s = 0.0
        self.global_steps = global_steps
        self.backend = backend

    def start(self):
        self._wall_start = time.perf_counter()

    def stop(self):
        if self._wall_start is not None:
            self.wall_total_s = time.perf_counter() - self._wall_start

    @contextmanager
    def segment(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.segments[key] += time.perf_counter() - t0

    @property
    def accounted_s(self) -> float:
        return sum(self.segments.values())

    @property
    def unaccounted_s(self) -> float:
        return self.wall_total_s - self.accounted_s

    def as_dict(self) -> dict:
        d = {
            "resync_orchestration": True,
            "global_steps": self.global_steps,
            "backend": self.backend,
            "wall_total_s": round(self.wall_total_s, 6),
        }
        for k in ORCH_KEYS:
            d[k] = round(self.segments[k], 6)
        d["accounted_s"] = round(self.accounted_s, 6)
        d["unaccounted_s"] = round(self.unaccounted_s, 6)
        if self.wall_total_s > 0:
            d["unaccounted_pct"] = round(100.0 * self.unaccounted_s / self.wall_total_s, 3)
            d["pct"] = {k: round(100.0 * self.segments[k] / self.wall_total_s, 3) for k in ORCH_KEYS}
        return d

    def emit(self):
        """Log one JSON line from the driver process."""
        try:
            logger.warning("RESYNC_ORCHESTRATION %s", json.dumps(self.as_dict(), sort_keys=True))
        except Exception as exc:  # never let instrumentation break a run
            logger.warning("RESYNC_ORCHESTRATION emit failed: %r", exc)


def maybe_new_orchestration(
    global_steps: int | None = None, backend: str | None = None
) -> "OrchestrationBreakdown | None":
    """Return a fresh orchestration accumulator when enabled, else ``None``."""
    if not is_enabled():
        return None
    return OrchestrationBreakdown(global_steps=global_steps, backend=backend)
