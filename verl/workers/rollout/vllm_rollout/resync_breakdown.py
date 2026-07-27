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
``s3_tensor_iter_s``   the whole per-tensor loop (pull from the param
                       generator, fill the bucket) INCLUDING the flush
                       spans below; per-tensor cost is s3 minus s4 minus s5
``s4_bucket_flush_s``  sum of the mid-loop bucket flushes: device
                       synchronize + send_pyobj + wait for receiver ack
``s5_final_flush_s``   the last bucket's synchronize + send + ack
``s6_cleanup_s``       socket close + buffer release + gc.collect +
                       ipc_collect + empty_cache
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
    "s3_tensor_iter_s",
    "s4_bucket_flush_s",
    "s5_final_flush_s",
    "s6_cleanup_s",
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
        """Loop time attributable to per-tensor work, by difference.

        ``s3`` brackets the whole iteration loop and therefore contains the
        mid-loop flushes (``s4``). Subtracting them leaves generator pull +
        bucket fill + metadata bookkeeping. Clamped at 0: the two spans are
        measured with the same clock but nesting means tiny negatives are
        possible on a pathologically short sync.
        """
        return max(0.0, self.segments["s3_tensor_iter_s"] - self.segments["s4_bucket_flush_s"])

    @property
    def accounted_s(self) -> float:
        # s4 is nested inside s3, so it must not be double counted.
        return (
            self.segments["s1_socket_init_s"]
            + self.segments["s2_buffer_init_s"]
            + self.segments["s3_tensor_iter_s"]
            + self.segments["s5_final_flush_s"]
            + self.segments["s6_cleanup_s"]
        )

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
        d["accounted_s"] = round(self.accounted_s, 6)
        d["unaccounted_s"] = round(self.unaccounted_s, 6)
        if self.wall_total_s > 0:
            d["unaccounted_pct"] = round(100.0 * self.unaccounted_s / self.wall_total_s, 3)
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
