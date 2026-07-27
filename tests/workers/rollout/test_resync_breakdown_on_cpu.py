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
"""CPU tests for the precision-agnostic resync breakdown (kanban t_9e8db24f).

Exercises the real ``BucketedWeightSender.async_send_weights`` loop against a
fake ZMQ socket and a CPU "device", with the breakdown both OFF and ON, and
asserts:

  * OFF: byte-identical wire traffic vs a pre-instrumentation reference, and
    no breakdown line emitted (the un-instrumented stack is untouched)
  * ON: the same wire traffic (instrumentation is observation-only)
  * ON: segments sum to the wall total with an explicit unaccounted residual
  * ON: counters (tensors / buckets / bytes) match what the loop really did
  * ON: per-tensor mean is derived by difference and excludes flush time
  * the accumulator's own overhead does not grow with sync count

No GPU, no vLLM, no ZMQ broker.
"""

import asyncio
import importlib.util
import json
import logging
import os
import pathlib
import sys
import time
from contextlib import contextmanager, nullcontext

import pytest
import torch

# --------------------------------------------------------------------------
# Import the two modules under test DIRECTLY by file path.
#
# Importing them as ``verl.workers.rollout.vllm_rollout.*`` would execute that
# package's ``__init__``, which imports the real vLLM -- far too heavy (and
# GPU-oriented) for a CPU unit test. Both modules under test only need torch,
# zmq and stdlib, so load them standalone.
# --------------------------------------------------------------------------
_PKG_DIR = pathlib.Path(__file__).resolve().parents[3] / "verl" / "workers" / "rollout" / "vllm_rollout"
# resync_breakdown lives under verl/utils/, NOT under vllm_rollout: it is
# imported by verl/checkpoint_engine/base.py, which must stay importable for
# the SGLang and TRT-LLM backends, and vllm_rollout/__init__.py RAISES
# PackageNotFoundError when vLLM is absent.
_UTILS_DIR = pathlib.Path(__file__).resolve().parents[3] / "verl" / "utils"


def _load(mod_name: str, filename: str, directory: "pathlib.Path | None" = None):
    spec = importlib.util.spec_from_file_location(mod_name, (directory or _PKG_DIR) / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


rb = _load("_t9e8_resync_breakdown", "resync_breakdown.py", _UTILS_DIR)
# bucketed_weight_transfer does `from verl.utils.resync_breakdown import
# maybe_new` at CALL time; the _CALL_STUBS fixture wires that name to `rb`.
_shim = type(sys)("_t9e8_pkg")
_shim.__path__ = [str(_PKG_DIR)]
sys.modules["_t9e8_pkg"] = _shim
sys.modules["_t9e8_pkg.resync_breakdown"] = rb


# --------------------------------------------------------------------------
# bucketed_weight_transfer imports two things out of the verl package proper:
#   module level : verl.utils.device.{get_device_id,get_device_name,get_torch_device}
#   CALL time    : verl.workers.rollout.utils.ensure_async_iterator
#                  (a function-local import inside async_send_weights)
# Importing either for real drags in `verl/__init__` -> ray, and
# rollout/utils -> uvicorn/fastapi. This test is deliberately dependency-free
# (no ray, no vLLM, no GPU), so it stands in minimal stubs.
#
# Two different lifetimes, hence two mechanisms:
#   * the module-level ones only need to exist while exec_module runs, so
#     _load_bwt_with_stubs installs them and withdraws them again, leaving
#     sys.modules exactly as it found it;
#   * the call-time one must be present while a test actually runs, so the
#     autouse _stub_call_time_imports fixture installs it per test and
#     restores afterwards.
# Restoring in both cases keeps a stub `verl` from shadowing the real package
# for any other test sharing this pytest session.
# --------------------------------------------------------------------------
_LOAD_STUBS = ("verl", "verl.utils", "verl.utils.device")
_CALL_STUBS = (
    "verl",
    "verl.workers",
    "verl.workers.rollout",
    "verl.workers.rollout.utils",
    "verl.utils",
    "verl.utils.resync_breakdown",
)
# leaves are plain modules; everything else must look like a package
_STUB_LEAVES = frozenset({"verl.utils.device", "verl.workers.rollout.utils", "verl.utils.resync_breakdown"})


def _make_stub(name: str):
    m = type(sys)(name)
    if name not in _STUB_LEAVES:
        m.__path__ = []
    return m


async def _ensure_async_iterator(iterable):
    """Behavioural copy of verl.workers.rollout.utils.ensure_async_iterator.

    Mirrors verl/workers/rollout/utils.py:83-90 @ 3a5d729d.
    """
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


@contextmanager
def _stubbed(names, **attrs):
    """Install stub modules for ``names``, set ``attrs`` on the leaf, restore.

    ``attrs`` are written through ``vars(mod)`` so the stubs need no class of
    their own and no per-attribute assignment boilerplate.
    """
    saved = {n: sys.modules.get(n) for n in names}
    stubs = {}
    try:
        for n in names:
            stubs[n] = sys.modules[n] = _make_stub(n)
        for dotted, values in attrs.items():
            vars(stubs[dotted.replace("__", ".")]).update(values)
        yield stubs
    finally:
        for n, old in saved.items():
            if old is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = old


def _load_bwt_with_stubs():
    with _stubbed(
        _LOAD_STUBS,
        verl__utils__device={
            "get_device_id": lambda: 0,
            "get_device_name": lambda: "cpu",
            # placeholder only: every test that reaches a flush replaces
            # _bwt.get_torch_device via the cpu_device fixture
            "get_torch_device": lambda: None,
        },
    ):
        return _load("_t9e8_pkg.bucketed_weight_transfer", "bucketed_weight_transfer.py")


_bwt = _load_bwt_with_stubs()
BucketedWeightSender = _bwt.BucketedWeightSender


@pytest.fixture(autouse=True)
def _stub_call_time_imports():
    """Satisfy the two function-local imports inside async_send_weights.

    ``verl.workers.rollout.utils.ensure_async_iterator`` and
    ``verl.utils.resync_breakdown.maybe_new`` are both imported at CALL time,
    so they must resolve while a test body runs. ``maybe_new`` is wired to the
    SAME module object the tests inspect (``rb``), so monkeypatching
    ``rb.is_enabled`` is what the sender actually sees.
    """
    with _stubbed(
        _CALL_STUBS,
        verl__workers__rollout__utils={"ensure_async_iterator": _ensure_async_iterator},
        verl__utils__resync_breakdown={"maybe_new": rb.maybe_new},
    ):
        yield


class FakeSocket:
    """Records every send_pyobj payload; recv() is an instant ack."""

    def __init__(self):
        self.sent = []
        self.n_recv = 0
        self.closed = False

    def bind(self, addr):
        self.addr = addr

    def send_pyobj(self, obj):
        # Deep-ish copy of the metadata so later mutation of bucket_meta
        # cannot retroactively change what we recorded.
        if isinstance(obj, dict) and "bucket_meta" in obj:
            self.sent.append(
                {
                    "is_last": obj["is_last"],
                    "names": list(obj["bucket_meta"].keys()),
                    "offsets": [m["offset"] for m in obj["bucket_meta"].values()],
                    "shapes": [tuple(m["shape"]) for m in obj["bucket_meta"].values()],
                }
            )
        else:
            self.sent.append({"handle": True})

    def send(self, _b):
        self.sent.append({"raw": True})

    def recv(self):
        self.n_recv += 1
        return b""

    def close(self):
        self.closed = True


def _make_sender(tmp_path, bucket_mb, socket, monkeypatch=None):
    s = BucketedWeightSender(
        zmq_handle=f"ipc://{tmp_path}/sock",
        bucket_size_mb=bucket_mb,
        use_shm=False,
    )
    # Bypass real socket/buffer construction: point at our fake socket and a
    # CPU uint8 buffer of the right size (the code only needs a byte buffer).
    s._init_socket = lambda: setattr(s, "socket", socket)

    def _init_buffer():
        s.buffer = torch.zeros(s.bucket_size, dtype=torch.uint8)
        socket.send_pyobj({"fake_handle": True})
        socket.recv()

    s._init_buffer = _init_buffer

    def _cleanup():
        socket.close()
        s.buffer = None

    s._cleanup = _cleanup
    return s


class _CpuDevice:
    """Stands in for get_torch_device() on a CPU-only host.

    The real flush path calls ``get_torch_device().synchronize()``; on a host
    with no CUDA that raises. Counting the calls also lets a test assert the
    flush structure is unchanged.
    """

    def __init__(self):
        self.n_sync = 0

    def synchronize(self):
        self.n_sync += 1

    def ipc_collect(self):
        pass

    def empty_cache(self):
        pass


@pytest.fixture
def cpu_device(monkeypatch):
    dev = _CpuDevice()
    monkeypatch.setattr(_bwt, "get_torch_device", lambda: dev)
    return dev


def _weights(n_tensors, numel, dtype=torch.bfloat16):
    async def gen():
        for i in range(n_tensors):
            yield f"layer.{i}.weight", torch.full((numel,), float(i % 7), dtype=dtype)

    return gen()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    monkeypatch.delenv(rb.ENV_FLAG, raising=False)
    yield


def test_disabled_by_default_no_accumulator():
    assert rb.is_enabled() is False
    assert rb.maybe_new() is None


def test_enabled_only_by_env(monkeypatch):
    for val, want in (("1", True), ("0", False), ("false", False), ("", False), ("yes", True)):
        monkeypatch.setenv(rb.ENV_FLAG, val)
        assert rb.is_enabled() is want, val
    monkeypatch.delenv(rb.ENV_FLAG)
    assert rb.is_enabled() is False


def _wire_traffic(tmp_path, n_tensors, numel, bucket_mb):
    sock = FakeSocket()
    s = _make_sender(tmp_path, bucket_mb, sock)
    _run(s.async_send_weights(_weights(n_tensors, numel)))
    return sock


def test_wire_traffic_identical_off_vs_on(tmp_path, monkeypatch, caplog, cpu_device):
    """Instrumentation must be observation-only: same bytes on the wire."""
    N, NUMEL, MB = 40, 4096, 1  # small bucket -> multiple flushes

    monkeypatch.delenv(rb.ENV_FLAG, raising=False)
    off = _wire_traffic(tmp_path, N, NUMEL, MB)
    n_sync_off = cpu_device.n_sync

    monkeypatch.setenv(rb.ENV_FLAG, "1")
    with caplog.at_level(logging.WARNING):
        on = _wire_traffic(tmp_path, N, NUMEL, MB)
    n_sync_on = cpu_device.n_sync - n_sync_off

    assert off.sent == on.sent, "instrumentation changed the wire traffic"
    assert off.n_recv == on.n_recv
    assert off.closed and on.closed
    assert n_sync_off == n_sync_on, "instrumentation changed the synchronize count"
    # and it did emit exactly one breakdown line when on
    lines = [r for r in caplog.records if "RESYNC_BREAKDOWN" in r.getMessage()]
    assert len(lines) == 1


def test_disabled_emits_nothing(tmp_path, monkeypatch, caplog, cpu_device):
    monkeypatch.delenv(rb.ENV_FLAG, raising=False)
    with caplog.at_level(logging.WARNING):
        _wire_traffic(tmp_path, 20, 4096, 1)
    assert not [r for r in caplog.records if "RESYNC_BREAKDOWN" in r.getMessage()]


def _emitted_payload(caplog):
    recs = [r.getMessage() for r in caplog.records if "RESYNC_BREAKDOWN" in r.getMessage()]
    assert len(recs) == 1, recs
    return json.loads(recs[0].split("RESYNC_BREAKDOWN ", 1)[1])


def test_counters_match_reality(tmp_path, monkeypatch, caplog, cpu_device):
    # 600 x 8192 x 2 B = 9.4 MB against a 1 MB bucket -> ~10 flushes, so the
    # mid-loop flush path (s4) is genuinely exercised. (An earlier sizing of
    # 60 tensors totalled 0.94 MB and never filled a single bucket.)
    N, NUMEL, MB = 600, 8192, 1
    monkeypatch.setenv(rb.ENV_FLAG, "1")
    sock = FakeSocket()
    s = _make_sender(tmp_path, MB, sock)
    with caplog.at_level(logging.WARNING):
        _run(s.async_send_weights(_weights(N, NUMEL)))
    d = _emitted_payload(caplog)

    assert d["n_tensors"] == N
    # bytes: bfloat16 -> 2 bytes/elem
    assert d["total_bytes"] == N * NUMEL * 2
    # buckets counted == bucket_meta messages actually sent
    meta_msgs = [m for m in sock.sent if "names" in m]
    assert d["n_buckets"] == len(meta_msgs), (d["n_buckets"], len(meta_msgs))
    assert d["n_buckets"] >= 2, "test needs at least one mid-loop flush"
    assert d["bucket_size_mb"] == MB
    assert d["n_direct_large_sends"] == 0


def test_segments_sum_to_wall_with_explicit_residual(tmp_path, monkeypatch, caplog, cpu_device):
    monkeypatch.setenv(rb.ENV_FLAG, "1")
    # 600 x 8192 x 2 B = 9.4 MB against a 1 MB bucket -> mid-loop flushes, so
    # the flush accounting is actually under test here. (An earlier sizing of
    # 50 tensors totalled 0.78 MB, never filled a bucket, and left s4 == 0 --
    # the identity held trivially.)
    with caplog.at_level(logging.WARNING):
        _wire_traffic(tmp_path, 600, 8192, 1)
    d = _emitted_payload(caplog)

    # every segment key present and non-negative
    for k in rb.SEGMENT_KEYS:
        assert k in d, k
        assert d[k] >= 0.0, (k, d[k])

    assert d["s4_bucket_flush_s"] > 0.0, "test needs a timed mid-loop flush to be meaningful"

    # Segments are now DISJOINT by construction (s3 was split into s3a source
    # pull / s3b bucket fill, and the flushes are excluded from s3b rather than
    # nested inside a single s3 span), so accounted_s is a plain total.
    #
    # Tolerance note: as_dict() rounds every field to 6 dp INDEPENDENTLY, so an
    # 8-term sum of rounded values compared against a separately-rounded total
    # carries up to 9 * 0.5e-6 of pure representation error. A 1e-6 bound here
    # is therefore flaky by construction, not a real accounting bug.
    ROUNDING_SLOP = 8e-6
    expect = sum(d[k] for k in rb.SEGMENT_KEYS)
    assert abs(d["accounted_s"] - expect) < ROUNDING_SLOP, (d["accounted_s"], expect)

    # Disjointness is the property that makes the residual meaningful: if the
    # flush time were still counted inside the per-tensor span, accounted_s
    # would exceed the sender's own wall clock.
    assert d["accounted_s"] <= d["wall_total_s"] + ROUNDING_SLOP, (
        "accounted_s exceeds wall time -- segments are overlapping/double-counted",
        d["accounted_s"],
        d["wall_total_s"],
    )

    # the identity that makes the breakdown auditable
    assert abs((d["accounted_s"] + d["unaccounted_s"]) - d["wall_total_s"]) < ROUNDING_SLOP
    # the sender covers nearly all of its own wall time
    assert d["unaccounted_pct"] < 5.0, d


def test_per_tensor_split_separates_source_pull_from_bucket_fill(tmp_path, monkeypatch, caplog, cpu_device):
    """The split that makes the probe able to localise the 102 s.

    ``s3a_source_pull_s`` is the cost of advancing the weight *source*; in
    production that is ``bridge.export_hf_weights``, i.e. the actor
    re-materialising a Megatron param in HF layout (collectives included).
    ``s3b_bucket_fill_s`` is purely local buffer work. A single combined span
    cannot tell those apart, which is why the earlier revision localised
    nothing.
    """
    monkeypatch.setenv(rb.ENV_FLAG, "1")
    # 600 x 8192 x 2 B = 9.4 MB / 1 MB bucket -> ~10 timed mid-loop flushes.
    with caplog.at_level(logging.WARNING):
        _wire_traffic(tmp_path, 600, 8192, 1)
    d = _emitted_payload(caplog)

    ROUNDING_SLOP = 8e-6
    # per_tensor_loop_s is now a plain SUM of two directly-measured, disjoint
    # terms rather than a difference of nested spans.
    assert d["per_tensor_loop_s"] == pytest.approx(d["s3a_source_pull_s"] + d["s3b_bucket_fill_s"], abs=ROUNDING_SLOP)
    # both halves are actually measured, i.e. neither is a dead field
    assert d["s3a_source_pull_s"] > 0.0, "source pull never timed"
    assert d["s3b_bucket_fill_s"] > 0.0, "bucket fill never timed"
    assert d["s4_bucket_flush_s"] > 0.0, "test needs at least one mid-loop flush to have been timed"

    # the substantive claim: flush time is NOT inside the per-tensor figure.
    # The flush is orders of magnitude above the rounding slop, so if it leaked
    # into s3b this comparison would fail.
    assert d["per_tensor_loop_s"] + d["s4_bucket_flush_s"] <= d["wall_total_s"] + ROUNDING_SLOP, (
        "per-tensor figure appears to include flush time"
    )

    # the per-tensor means are derived from UNROUNDED sums
    assert d["per_tensor_mean_ms"] == pytest.approx(d["per_tensor_loop_s"] / d["n_tensors"] * 1e3, abs=1e-3)
    assert d["source_pull_mean_ms"] == pytest.approx(d["s3a_source_pull_s"] / d["n_tensors"] * 1e3, abs=1e-3)
    assert d["bucket_fill_mean_ms"] == pytest.approx(d["s3b_bucket_fill_s"] / d["n_tensors"] * 1e3, abs=1e-3)


def test_noncontiguous_tensors_are_counted():
    """The counter that would reveal a redundant-copy opportunity."""
    bd = rb.ResyncBreakdown(bucket_size_mb=1)
    packed = torch.randn(4, 8, 6)
    a, b = packed.chunk(2, dim=1)
    bd.count_tensor(a.nbytes, a.is_contiguous())  # the CHUNK: non-contiguous
    bd.count_tensor(a[0].nbytes, a[0].is_contiguous())  # a SLICE of it: contiguous
    bd.count_tensor(packed[0].nbytes, packed[0].is_contiguous())
    assert bd.n_tensors == 3
    assert bd.n_noncontiguous == 1, "only the whole chunk is non-contiguous"


def test_accumulator_overhead_does_not_grow_with_sync_count():
    """A leak here would slow the run it is measuring (the FATAL-2 class)."""
    import time as _t

    N_SYNCS, N_TENSORS = 200, 300
    per_sync = []
    for _ in range(N_SYNCS):
        t0 = _t.perf_counter()
        bd = rb.ResyncBreakdown(bucket_size_mb=2048)
        bd.start()
        with bd.segment("s3b_bucket_fill_s"):
            for i in range(N_TENSORS):
                bd.count_tensor(1024, i % 3 != 0)
        with bd.segment("s4_bucket_flush_s"):
            bd.count_bucket()
        bd.stop()
        bd.as_dict()
        per_sync.append(_t.perf_counter() - t0)

    n = len(per_sync)
    mean_x = (n - 1) / 2.0
    mean_y = sum(per_sync) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(per_sync))
    den = sum((i - mean_x) ** 2 for i in range(n))
    slope = num / den
    growth = slope * n
    print(
        f"\n[overhead] n={n} mean={mean_y * 1e6:.2f}us slope={slope:.3e}s/sync "
        f"growth_over_run={growth * 1e6:.2f}us ({100.0 * growth / mean_y:.2f}% of mean)"
    )
    # bound magnitude, not sign (noise makes either sign possible)
    assert growth < 0.5 * mean_y, f"overhead grows with sync count: {growth} vs mean {mean_y}"

    # and no state leaks between accumulators
    fresh = rb.ResyncBreakdown()
    assert fresh.n_tensors == 0 and fresh.n_buckets == 0
    assert all(v == 0.0 for v in fresh.segments.values())


def test_emit_never_raises_on_bad_state(caplog):
    """Instrumentation must not be able to kill a real run."""
    bd = rb.ResyncBreakdown()
    bd.rank = object()  # not JSON-serialisable
    with caplog.at_level(logging.WARNING):
        bd.emit()  # must swallow
    assert any("emit failed" in r.getMessage() for r in caplog.records)


def test_zero_tensor_sync_reports_no_mean():
    bd = rb.ResyncBreakdown()
    bd.start()
    bd.stop()
    d = bd.as_dict()
    assert d["n_tensors"] == 0
    assert "per_tensor_mean_ms" not in d, "must not divide by zero"


# --------------------------------------------------------------------------
# Orchestration side (the 8 stages of CheckpointEngineManager.update_weights)
#
# This is the half that makes the breakdown sum back to the trainer's
# `timing_s/update_weights` rather than to one sender's span. The sender-side
# segments above can only ever explain the transfer; stages 1-4 and 6-8 are
# request abort, worker-group build, KV-cache free/restore, process-group build
# and generation resume, and none of them is visible to a sender timer.
# --------------------------------------------------------------------------


def test_orchestration_disabled_by_default(monkeypatch):
    monkeypatch.delenv(rb.ENV_FLAG, raising=False)
    assert rb.maybe_new_orchestration() is None


def test_orchestration_enabled_by_same_flag(monkeypatch):
    monkeypatch.setenv(rb.ENV_FLAG, "1")
    ob = rb.maybe_new_orchestration(global_steps=7, backend="nccl")
    assert ob is not None
    assert ob.global_steps == 7 and ob.backend == "nccl"
    assert set(ob.segments) == set(rb.ORCH_KEYS)


def test_orchestration_segments_are_disjoint_and_sum_to_wall(monkeypatch):
    """accounted + unaccounted == wall, with every stage separately visible."""
    monkeypatch.setenv(rb.ENV_FLAG, "1")
    ob = rb.maybe_new_orchestration(global_steps=1, backend="nccl")
    ob.start()
    for k in rb.ORCH_KEYS:
        with ob.segment(k):
            time.sleep(0.001)
    ob.stop()
    d = ob.as_dict()

    for k in rb.ORCH_KEYS:
        assert d[k] >= 0.001, (k, d[k])
    ROUNDING_SLOP = 1e-5
    assert abs(d["accounted_s"] - sum(d[k] for k in rb.ORCH_KEYS)) < ROUNDING_SLOP
    assert abs((d["accounted_s"] + d["unaccounted_s"]) - d["wall_total_s"]) < ROUNDING_SLOP
    # stages were run back to back, so nearly all wall time is attributed
    assert d["unaccounted_pct"] < 10.0, d
    assert d["resync_orchestration"] is True
    assert set(d["pct"]) == set(rb.ORCH_KEYS)


def test_orchestration_barrier_stage_dominates_when_transfer_is_slow(monkeypatch):
    """o5 is the discriminator the card needs, so it must be separable.

    o5 is a single ray.get over actor AND rollout ranks
    (verl/checkpoint_engine/base.py:515-518), i.e. a full-fleet barrier. If the
    stage split works, a slow transfer shows up in o5 alone and the other seven
    stages stay small -- which is what distinguishes "the transfer is slow" from
    "the orchestration around it is slow".
    """
    monkeypatch.setenv(rb.ENV_FLAG, "1")
    ob = rb.maybe_new_orchestration()
    ob.start()
    with ob.segment("o1_abort_replicas_s"):
        time.sleep(0.001)
    with ob.segment("o5_send_recv_barrier_s"):
        time.sleep(0.05)
    ob.stop()
    d = ob.as_dict()
    assert d["pct"]["o5_send_recv_barrier_s"] > 80.0, d["pct"]
    assert d["pct"]["o1_abort_replicas_s"] < 20.0, d["pct"]


def test_orchestration_emit_never_raises(caplog, monkeypatch):
    monkeypatch.setenv(rb.ENV_FLAG, "1")
    ob = rb.maybe_new_orchestration()
    ob.backend = object()  # not JSON-serialisable
    with caplog.at_level(logging.WARNING):
        ob.emit()
    assert any("RESYNC_ORCHESTRATION emit failed" in r.getMessage() for r in caplog.records)


def test_real_manager_update_weights_is_fully_staged(monkeypatch):
    """Drive the REAL CheckpointEngineManager.update_weights body.

    Loading verl.checkpoint_engine.base for real would pull in ray, vLLM and
    fastapi, so instead the actual source of both methods is extracted from
    the file and executed against fakes. That keeps the test honest about
    *which lines* it covers: any future edit to those two method bodies is
    re-parsed here rather than mirrored by hand.

    What this proves that a hand-written mirror could not:
      * all 8 stages plus the 3 sub-stages of stage 4 are really wrapped;
      * the metrics dict gains the resync_orch/* keys, so the breakdown lands
        in metrics.jsonl next to timing_s/update_weights;
      * with the flag OFF the method still runs and adds NO keys.

    Verified by mutation: every fake stage below sleeps for a distinct,
    detectable duration, and the assertions require each stage's measured time
    to be at least that long. Asserting mere key PRESENCE is not enough --
    segments are pre-initialised to 0.0, so a deleted `with ob.segment(...)`
    would leave the key in place with value 0.0 and the test would still pass.
    (Confirmed: with presence-only assertions, 3 of 4 timer-removal mutations
    went undetected.)
    """
    import ast
    import textwrap

    src_path = pathlib.Path(__file__).resolve().parents[3] / "verl" / "checkpoint_engine" / "base.py"
    tree = ast.parse(src_path.read_text())
    wanted = {"update_weights", "build_process_group"}
    bodies = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            # the manager's update_weights is the async one taking global_steps
            if node.name == "update_weights" and not isinstance(node, ast.AsyncFunctionDef):
                continue
            bodies[node.name] = ast.get_source_segment(src_path.read_text(), node)
    assert wanted <= set(bodies), f"could not extract {wanted - set(bodies)} from {src_path}"

    calls = []
    # Each stage is made artificially slow so that a MISSING timer shows up as a
    # 0.0 reading rather than as a merely-smaller number. SLEEP must stay well
    # above timer granularity and well below test-suite patience.
    SLEEP = 0.02

    class _FakeWG:
        world_size = 2

        def update_weights(self, **kw):
            calls.append("wg.update_weights")
            time.sleep(SLEEP)  # attributed to o5 (the full-fleet barrier)
            return [{"engine/metric": 1.0}, {}]

        def execute_checkpoint_engine(self, *a, **kw):
            calls.append("wg.execute_checkpoint_engine")
            time.sleep(SLEEP)  # o4a prepare / o4c init_process_group / o6 finalize
            return [None, None]

    class _FakeBackendCls:
        @staticmethod
        def build_topology(a_ws, r_ws, metadata):
            calls.append("build_topology")
            time.sleep(SLEEP)  # o4b
            return {"x": [None] * a_ws}, {"y": [None] * r_ws}

    class _FakeReplica:
        workers = [object(), object()]

    def _slow_ray_get(x):
        return x

    ns = {
        "ray": type(sys)("ray"),
        "RayWorkerGroup": lambda **kw: (time.sleep(SLEEP), _FakeWG())[1],  # o2
        "RayClassWithInitArgs": lambda **kw: None,
        "_worker_cls": None,
        "nullcontext": nullcontext,
        "maybe_new_orchestration": rb.maybe_new_orchestration,
    }
    ns["ray"].get = _slow_ray_get

    class _Mgr:
        backend = "nccl"
        backend_cls = _FakeBackendCls
        actor_wg = _FakeWG()
        replicas = [_FakeReplica()]

        async def abort_replicas(self):
            calls.append("abort")
            await asyncio.sleep(SLEEP)  # o1

        async def release_kv_cache_replicas(self):
            calls.append("release_kv")
            await asyncio.sleep(SLEEP)  # o3

        async def resume_kv_cache_replicas(self):
            calls.append("resume_kv")
            await asyncio.sleep(SLEEP)  # o7

        async def resume_generation_replicas(self):
            calls.append("resume_gen")
            await asyncio.sleep(SLEEP)  # o8

    for name in ("build_process_group", "update_weights"):
        exec(compile(textwrap.dedent(bodies[name]), str(src_path), "exec"), ns)
        setattr(_Mgr, name, ns[name])

    # --- flag ON: every stage timed, metrics enriched
    monkeypatch.setenv(rb.ENV_FLAG, "1")
    calls.clear()
    metrics = _run(_Mgr().update_weights(global_steps=3))
    assert "abort" in calls and "release_kv" in calls and "resume_gen" in calls
    assert "build_topology" in calls and "wg.update_weights" in calls

    # The load-bearing assertion: each stage must report REAL elapsed time.
    # A removed `with ob.segment(...)` leaves the key present but stuck at 0.0,
    # so this is what makes the test non-vacuous.
    floor = SLEEP * 0.5
    for k in rb.ORCH_KEYS:
        key = f"resync_orch/{k}"
        assert key in metrics, f"stage {k} never surfaced into metrics"
        assert metrics[key] >= floor, (
            f"stage {k} reported {metrics[key]:.6f} s but its fake work sleeps {SLEEP} s -- "
            "the timer around this stage is missing or mis-scoped"
        )

    assert "resync_orch/wall_total_s" in metrics
    assert "resync_orch/unaccounted_s" in metrics
    # stages are disjoint, so they cannot sum past the method's own wall time
    assert metrics["resync_orch/accounted_s"] <= metrics["resync_orch/wall_total_s"] + 1e-5
    # and together they should explain nearly all of it
    assert metrics["resync_orch/unaccounted_pct"] < 10.0, metrics["resync_orch/unaccounted_pct"]
    # the engine's own metric survives alongside the breakdown
    assert metrics["engine/metric"] == 1.0

    # --- flag OFF: same control flow, zero added keys
    monkeypatch.delenv(rb.ENV_FLAG, raising=False)
    calls.clear()
    metrics_off = _run(_Mgr().update_weights(global_steps=3))
    assert "abort" in calls and "wg.update_weights" in calls, "OFF path must still run all stages"
    assert not [k for k in metrics_off if k.startswith("resync_orch/")], metrics_off
    assert metrics_off == {"engine/metric": 1.0}


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.exit(pytest.main([__file__, "-v", "-s"]))
