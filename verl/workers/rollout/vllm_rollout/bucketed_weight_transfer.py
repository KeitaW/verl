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
"""
Bucketed weight transfer via ZMQ + IPC (or shared memory fallback).

Not recommended depending on vllm for this file.
"""

import gc
import logging
import os
from contextlib import nullcontext
from multiprocessing import shared_memory
from typing import Callable, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_device_id, get_device_name, get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int
    handle: tuple


# copy from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


def create_shared_memory(size: int, name: str):
    """Create shared memory for weight transfer. If already exists, attach to it."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        assert shm.size >= size, f"Stale shm segment '{name}': expected {size} bytes, got {shm.size}"
    return shm


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    """Rebuild tensor from shared memory."""
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)

    return tensor, shm


class BucketedWeightSender:
    """
    Send model weights via bucketed IPC transfer over ZMQ.

    Packs weight tensors into a fixed-size communication buffer and sends them
    in buckets to the receiver. Supports CUDA IPC and shared memory fallback.

    Args:
        zmq_handle: ZMQ IPC socket path (e.g., "ipc:///tmp/rl-colocate-zmq-<uuid>.sock")
        bucket_size_mb: Communication buffer size in MB
        use_shm: Use shared memory instead of CUDA IPC (for NPU compatibility)
    """

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    async def async_send_weights(self, weights):
        """
        Send weights to the receiver. Accepts a sync generator or async iterator.

        Args:
            weights: Generator or async iterator yielding (name, tensor) pairs
        """
        from verl.utils.resync_breakdown import maybe_new
        from verl.workers.rollout.utils import ensure_async_iterator

        # Off unless VERL_RESYNC_BREAKDOWN=1; disabled costs one env lookup and
        # leaves every code path below byte-identical in behaviour.
        bd = maybe_new(bucket_size_mb=self.bucket_size_mb)
        if bd is not None:
            bd.start()
        try:
            if bd is None:
                self._init_socket()
                self._init_buffer()
            else:
                with bd.segment("s1_socket_init_s"):
                    self._init_socket()
                with bd.segment("s2_buffer_init_s"):
                    self._init_buffer()

            # send bucket weights
            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            # dtype = PrecisionType.to_dtype(self.config.dtype)
            #
            # NOTE(kanban t_9e8db24f): the SOURCE PULL is timed separately from
            # the bucket fill (s3a vs s3b), which is the whole point of this
            # probe rather than an optional refinement.
            #
            # In production `weights` is not a cheap iterator: it is
            # `bridge.export_hf_weights(self.module)`
            # (verl/workers/engine/megatron/transformer_impl.py:826 @ 3a5d729d),
            # so advancing it re-materialises each Megatron parameter in HF
            # layout -- for a TP/PP/EP-sharded actor feeding a differently
            # sharded rollout, that involves collectives per tensor. Bracketing
            # the loop as one span would therefore lump "the actor gathered the
            # weight" together with "we copied it into the bucket" and localise
            # nothing.
            #
            # The split costs two clock reads per tensor. Measured on this
            # host: 36,945 tensors -> +6.18 ms, i.e. 0.006% of the 102.60 s
            # update_weights median (workspace clock-overhead.json). That is
            # three orders of magnitude below the signal, so the split is taken
            # unconditionally rather than hidden behind a second flag.
            aiter = ensure_async_iterator(weights).__aiter__()
            while True:
                if bd is None:
                    try:
                        name, weight = await aiter.__anext__()
                    except StopAsyncIteration:
                        break
                else:
                    _t0 = bd.clock()
                    try:
                        name, weight = await aiter.__anext__()
                    except StopAsyncIteration:
                        # the final pull still costs whatever it costs
                        bd.add("s3a_source_pull_s", bd.clock() - _t0)
                        break
                    bd.add("s3a_source_pull_s", bd.clock() - _t0)
                    _t1 = bd.clock()

                # model parameters are in fp32 full precision
                # (vermouth1992) we should not force cast weight here because some parameters
                # (such as moe gate) have to keep fp32 precision. If a weight is bf16 in the rollout side,
                # the rollout should automatically cast on demand. However, this would incur a higher weight
                # transfer volume.
                # weight = weight.to(dtype, non_blocking=True)
                if bd is not None:
                    bd.count_tensor(weight.nbytes, weight.is_contiguous())

                # fill the tensor bucket
                if offset + weight.nbytes > self.bucket_size and len(bucket_meta) > 0:
                    if bd is None:
                        get_torch_device().synchronize()
                        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                        self.socket.recv()
                    else:
                        # exclude the flush from s3b so the two never overlap
                        bd.add("s3b_bucket_fill_s", bd.clock() - _t1)
                        with bd.segment("s4_bucket_flush_s"):
                            get_torch_device().synchronize()
                            self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                            self.socket.recv()
                        bd.count_bucket()
                        _t1 = bd.clock()
                    bucket_meta = {}
                    offset = 0

                if offset + weight.nbytes > self.bucket_size:
                    assert not self.use_shm, (
                        f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
                        f"Please increase rollout.update_weights_bucket_megabytes({self.bucket_size_mb} MB)."
                    )
                    if bd is not None:
                        bd.count_direct_large()
                        bd.add("s3b_bucket_fill_s", bd.clock() - _t1)
                        with bd.segment("s7_direct_large_s"):
                            self._direct_send_large_weight(name, weight)
                    else:
                        self._direct_send_large_weight(name, weight)
                    continue

                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                    "handle": None,
                }
                self.buffer[offset : offset + weight.nbytes].view(dtype=weight.dtype).view(weight.shape).copy_(
                    weight, non_blocking=True
                )
                offset += weight.nbytes
                if bd is not None:
                    bd.add("s3b_bucket_fill_s", bd.clock() - _t1)

            # send the last bucket
            final_ctx = bd.segment("s5_final_flush_s") if bd is not None else nullcontext()
            with final_ctx:
                get_torch_device().synchronize()
                self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
                self.socket.recv()
            if bd is not None:
                bd.count_bucket()
        finally:
            if bd is None:
                self._cleanup()
            else:
                with bd.segment("s6_cleanup_s"):
                    self._cleanup()
                bd.stop()
                bd.emit()

    def _init_socket(self):
        """Initialize ZMQ REQ socket and bind."""
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self):
        """build communication buffer"""
        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=f"{get_device_name()}:{get_device_id()}")
            handle = reduce_tensor(buffer)
            self.socket.send_pyobj(handle)
        else:
            import uuid

            # Create unique name for shared memory
            shm_name = f"verl_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(self.bucket_size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

            comm_metadata = {"name": shm_name, "size": self.bucket_size}
            self.socket.send_pyobj(comm_metadata)

        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()

    def _direct_send_large_weight(self, name: str, weight: torch.Tensor):
        """Send a weight larger than the bucket size via cuda ipc or share memory."""
        logger.debug(f"Direct sending large weight {name}({weight.shape}, {weight.dtype})")
        # TODO: support fallback to shared memory
        handle = reduce_tensor(weight)
        bucket_meta: dict[str, TensorMetadata] = {}
        bucket_meta[name] = {
            "name": name,
            "shape": weight.shape,
            "dtype": weight.dtype,
            "offset": 0,
            "handle": handle,
        }
        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
        self.socket.recv()


class BucketedWeightReceiver:
    """
    Receive model weights via bucketed IPC transfer over ZMQ.

    Receives weight tensors from BucketedWeightSender and passes each
    bucket to a callback for processing (e.g., loading into the model).

    Args:
        zmq_handle: ZMQ IPC socket path (must match sender)
        device: Target device for received tensors
        use_shm: Use shared memory instead of CUDA IPC
    """

    def __init__(
        self,
        zmq_handle: str,
        device: torch.device,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def receive_weights(self, on_bucket_received: callable):
        """
        Receive weights from sender and process each bucket via callback.

        Args:
            on_bucket_received: Callback function(weights: list[(name, tensor)]) called per bucket.
        """
        try:
            self._init_socket()
            self._init_buffer()

            # receive bucket and update weights
            while True:
                metadata = self.socket.recv_pyobj()
                weights, tensor = [], None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset, handle = meta["shape"], meta["dtype"], meta["offset"], meta["handle"]
                    if handle is not None:
                        tensor = rebuild_ipc(handle, self.device.index)
                        weights.append((name, tensor))
                        continue
                    size = dtype.itemsize * shape.numel()
                    tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    weights.append((name, tensor))
                on_bucket_received(weights)
                get_torch_device().synchronize()
                self.socket.send(b"")
                del weights, tensor
                if metadata["is_last"]:
                    break
        finally:
            self._cleanup()

    def iter_weights(self):
        """Yield received weights one-by-one while preserving bucket backpressure."""
        try:
            self._init_socket()
            self._init_buffer()

            while True:
                metadata = self.socket.recv_pyobj()
                tensor = None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset, handle = meta["shape"], meta["dtype"], meta["offset"], meta["handle"]
                    if handle is not None:
                        tensor = rebuild_ipc(handle, self.device.index)
                        yield name, tensor
                        continue
                    size = dtype.itemsize * shape.numel()
                    tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    yield name, tensor
                get_torch_device().synchronize()
                self.socket.send(b"")
                tensor = None
                if metadata["is_last"]:
                    break
        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ REP socket and connect."""
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self):
        """Receive and rebuild communication buffer from sender."""
        comm_metadata = self.socket.recv_pyobj()
        buffer, shm = None, None
        if not self.use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        # Synchronize before releasing the buffer to ensure all async ops
        # referencing it (e.g. clone, .to()) have completed.
        get_torch_device().synchronize()
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()
