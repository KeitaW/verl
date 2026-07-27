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
from contextlib import contextmanager

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


def _load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, _PKG_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


rb = _load("_t9e8_resync_breakdown", "resync_breakdown.py")
# bucketed_weight_transfer does `from .resync_breakdown import maybe_new`, a
# relative import; satisfy it by pre-registering a tiny package shim.
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
_CALL_STUBS = ("verl", "verl.workers", "verl.workers.rollout", "verl.workers.rollout.utils")
# leaves are plain modules; everything else must look like a package
_STUB_LEAVES = frozenset({"verl.utils.device", "verl.workers.rollout.utils"})


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
    """Satisfy the function-local `from verl.workers.rollout.utils import ...`."""
    with _stubbed(
        _CALL_STUBS,
        verl__workers__rollout__utils={"ensure_async_iterator": _ensure_async_iterator},
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
    # the s4-nested-in-s3 accounting is actually under test here. (An earlier
    # sizing of 50 tensors totalled 0.78 MB, never filled a bucket, and left
    # s4 == 0 -- the identity held trivially.)
    with caplog.at_level(logging.WARNING):
        _wire_traffic(tmp_path, 600, 8192, 1)
    d = _emitted_payload(caplog)

    # every segment key present and non-negative
    for k in rb.SEGMENT_KEYS:
        assert k in d, k
        assert d[k] >= 0.0, (k, d[k])

    assert d["s4_bucket_flush_s"] > 0.0, "test needs a timed mid-loop flush to be meaningful"

    # accounted must NOT double-count s4 (nested inside s3).
    #
    # Tolerance note: as_dict() rounds every field to 6 dp INDEPENDENTLY, so a
    # 5-term sum of rounded values compared against a separately-rounded total
    # carries up to 6 * 0.5e-6 = 3e-6 of pure representation error. A 1e-6
    # bound here is therefore flaky by construction (observed failing at
    # residual ~1e-6 on a fast host), not a real accounting bug. Bound the
    # rounding explicitly.
    ROUNDING_SLOP = 4e-6
    expect = (
        d["s1_socket_init_s"]
        + d["s2_buffer_init_s"]
        + d["s3_tensor_iter_s"]
        + d["s5_final_flush_s"]
        + d["s6_cleanup_s"]
    )
    assert abs(d["accounted_s"] - expect) < ROUNDING_SLOP, (d["accounted_s"], expect)

    # s4 must be excluded: adding it in would overshoot by the whole flush time,
    # which is orders of magnitude above the rounding slop.
    over = expect + d["s4_bucket_flush_s"]
    assert abs(d["accounted_s"] - over) > ROUNDING_SLOP, "s4 appears to be double-counted inside accounted_s"

    # the identity that makes the breakdown auditable
    assert abs((d["accounted_s"] + d["unaccounted_s"]) - d["wall_total_s"]) < ROUNDING_SLOP
    # the sender covers nearly all of its own wall time
    assert d["unaccounted_pct"] < 5.0, d


def test_per_tensor_mean_excludes_flush_time(tmp_path, monkeypatch, caplog, cpu_device):
    """s4 is nested in s3; the per-tensor figure must subtract it."""
    monkeypatch.setenv(rb.ENV_FLAG, "1")
    # 600 x 8192 x 2 B = 9.4 MB / 1 MB bucket -> ~10 timed mid-loop flushes.
    with caplog.at_level(logging.WARNING):
        _wire_traffic(tmp_path, 600, 8192, 1)
    d = _emitted_payload(caplog)

    # per_tensor_loop_s is (s3 - s4) computed on UNROUNDED values, then rounded;
    # the right-hand side is a difference of two SEPARATELY rounded fields. Both
    # sides therefore carry ~0.5e-6 of representation error, so an abs=1e-6
    # bound sits exactly on the boundary and fails ~12% of runs (measured: 3
    # failures in 25 repeats, all at residual 1e-6). Use the same explicit
    # rounding slop as the accounting test.
    ROUNDING_SLOP = 4e-6
    assert d["per_tensor_loop_s"] == pytest.approx(d["s3_tensor_iter_s"] - d["s4_bucket_flush_s"], abs=ROUNDING_SLOP)
    assert d["per_tensor_loop_s"] <= d["s3_tensor_iter_s"]
    assert d["s4_bucket_flush_s"] > 0.0, "test needs at least one mid-loop flush to have been timed"
    # the substantive claim: flush time is genuinely excluded, by a margin far
    # larger than the rounding slop
    assert d["per_tensor_loop_s"] < d["s3_tensor_iter_s"] - ROUNDING_SLOP, (
        "per-tensor figure does not appear to exclude flush time"
    )
    # per_tensor_mean_ms is derived from the UNROUNDED per_tensor_loop_s, so
    # compare against the accumulator's own value rather than the rounded
    # field echoed in the JSON.
    assert d["per_tensor_mean_ms"] == pytest.approx(d["per_tensor_loop_s"] / d["n_tensors"] * 1e3, abs=1e-3)


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
        with bd.segment("s3_tensor_iter_s"):
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


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.exit(pytest.main([__file__, "-v", "-s"]))
