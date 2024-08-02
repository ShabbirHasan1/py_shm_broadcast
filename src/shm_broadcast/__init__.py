# Do Checkout https://github.com/vllm-project/vllm #type: ignore
# Thanks to @vllm-project for the awesome project!
from __future__ import annotations

import logging
import os
import pickle
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Dict, List, Optional
from unittest.mock import patch

import torch
from zmq import SUB, SUBSCRIBE, XPUB, XPUB_VERBOSE, Context

__all__ = ["MessageQueue", "ShmRingBuffer", "Handle", "get_ip", "get_open_port"]

RINGBUFFER_WARNING_INTERVAL: float = 0.01
RINGBUFFER_SLEEP_INTERVAL = 1e-7

logger = logging.getLogger(__name__)


@staticmethod
def get_ip() -> str:
    # IP is not set, try to get it from the network interface
    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    logger.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " VLLM_HOST_IP or HOST_IP.",
        stacklevel=2,
    )
    return "0.0.0.0"


@staticmethod
def get_open_port() -> int:
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


class ShmRingBuffer:
    def __init__(
        self,
        n_reader: int,
        max_chunk_bytes: int,
        max_chunks: int,
        name: Optional[str] = None,
    ) -> None:
        """
        A shared memory ring buffer implementation for broadcast communication.
        Essentially, it is a queue where only one will `enqueue` and multiple will `dequeue`.
        The max size of each item, together with the max number of items that can be stored
        in the buffer are known in advance. In this case, we don't need to synchronize the access.
        """
        self.n_reader = n_reader
        self.metadata_size = 1 + n_reader
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        self.total_bytes_of_buffer = (
            self.max_chunk_bytes + self.metadata_size
        ) * self.max_chunks
        self.data_offset = 0
        self.metadata_offset = self.max_chunk_bytes * self.max_chunks
        logger.info("total_bytes_of_buffer: %s", self.total_bytes_of_buffer)
        if name is None:
            self.is_creator = True
            self.shared_memory = shared_memory.SharedMemory(
                create=True, size=self.total_bytes_of_buffer
            )
            # initialize the metadata section to 0
            with memoryview(
                self.shared_memory.buf[self.metadata_offset :]
            ) as metadata_buffer:
                torch.frombuffer(metadata_buffer, dtype=torch.uint8).fill_(0)
        else:
            # we are opening an existing buffer
            self.is_creator = False
            # fix to https://stackoverflow.com/q/62748654/9191338
            # Python incorrectly tracks shared memory even if it is not
            # created by the process. The following patch is a workaround.
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                try:
                    self.shared_memory = shared_memory.SharedMemory(name=name)
                except FileNotFoundError:
                    # we might deserialize the object in a different node
                    # in this case, this object is not used,
                    # and we should suppress the error
                    pass

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.n_reader,
                self.max_chunk_bytes,
                self.max_chunks,
                self.shared_memory.name,
            ),
        )

    def __del__(self):
        if hasattr(self, "shared_memory"):
            self.shared_memory.close()
            if self.is_creator:
                self.shared_memory.unlink()

    @contextmanager
    def get_data(self, current_idx: int):
        start = self.data_offset + current_idx * self.max_chunk_bytes
        end = start + self.max_chunk_bytes
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf

    @contextmanager
    def get_metadata(self, current_idx: int):
        start = self.metadata_offset + current_idx * self.metadata_size
        end = start + self.metadata_size
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf


@dataclass
class Handle:
    connect_ip: str
    local_reader_ranks: List[int] = field(default_factory=list)
    buffer: Optional[ShmRingBuffer] = None
    local_subscribe_port: Optional[int] = None
    remote_subscribe_port: Optional[int] = None


class MessageQueue:
    """A message queue that uses shared memory to communicate between processes.

    This class provides a message queue that uses shared memory to communicate
    between processes. It is designed to be used in a distributed computing
    environment where multiple processes need to communicate with each other.
    The shared memory ring buffer is used to store the data that is being
    communicated, and the publish-subscribe pattern is used to send messages
    between the processes.

    The shared memory ring buffer is implemented using the `multiprocessing`
    module, which allows multiple processes to access the same shared memory
    buffer. The shared memory ring buffer is created using the `multiprocessing`
    module, and the publish-subscribe pattern is implemented using the `zmq`
    library.

    The shared memory ring buffer is used to store the data that is being
    communicated. The data is stored in chunks, and each chunk is associated
    with a flag that indicates whether the chunk has been read or not. The
    shared memory ring buffer is divided into two parts: the metadata part and
    the data part. The metadata part contains the flags that indicate whether
    the chunk has been read or not. The data part contains the actual data.

    The publish-subscribe pattern is used to send messages between the processes.
    Each process has a publish-subscribe socket that is used to send messages to
    all other processes. The publish-subscribe socket is used to send messages
    to all other processes, and the subscribe-publish socket is used to receive
    messages from all other processes.

    The shared memory ring buffer is divided into two parts: the metadata part
    and the data part. The metadata part contains the flags that indicate whether
    the chunk has been read or not. The data part contains the actual data.

    The shared memory ring buffer is divided into two parts: the metadata part
    and the data part. The metadata part contains the flags that indicate whether
    the chunk has been read or not. The data part contains the actual data.
    """

    def __init__(
        self,
        n_reader,  # number of all readers
        n_local_reader,  # number of local readers through shared memory
        local_reader_ranks: Optional[List[int]] = None,
        max_chunk_bytes: int = 1024 * 1024 * 10,  # 10 MB
        max_chunks: int = 10,
        connect_ip: Optional[str] = None,
    ):
        if local_reader_ranks is None:
            local_reader_ranks = list(range(n_local_reader))
        else:
            assert len(local_reader_ranks) == n_local_reader
        self.n_local_reader = n_local_reader
        n_remote_reader = n_reader - n_local_reader
        self.n_remote_reader = n_remote_reader

        if connect_ip is None:
            connect_ip = get_ip() if n_remote_reader > 0 else "127.0.0.1"

        context = Context()

        if n_local_reader > 0:
            # for local readers, we will:
            # 1. create a shared memory ring buffer to communicate small data
            # 2. create a publish-subscribe socket to communicate large data
            self.buffer = ShmRingBuffer(
                n_reader=n_local_reader,
                max_chunk_bytes=max_chunk_bytes,
                max_chunks=max_chunks,
                name=None,
            )
            # XPUB is very similar to PUB,
            # except that it can receive subscription messages
            # to confirm the number of subscribers
            self.local_socket = context.socket(XPUB)
            # set the verbose option so that we can receive every subscription
            # message. otherwise, we will only receive the first subscription
            # see http://api.zeromq.org/3-3:zmq-setsockopt for more details
            self.local_socket.setsockopt(XPUB_VERBOSE, True)
            # local_subscribe_port = get_open_port()
            local_subscribe_port = 12345
            self.local_socket.bind(f"tcp://*:{local_subscribe_port}")

            self.current_idx = 0

        else:
            self.buffer = None  # type: ignore
            local_subscribe_port = None
            self.local_socket = None
            self.current_idx = -1

        if n_remote_reader > 0:
            # for remote readers, we will:
            # create a publish-subscribe socket to communicate large data
            self.remote_socket = context.socket(XPUB)
            self.remote_socket.setsockopt(XPUB_VERBOSE, True)
            # remote_subscribe_port = get_open_port()
            remote_subscribe_port = 12346
            self.remote_socket.bind(f"tcp://*:{remote_subscribe_port}")

        else:
            remote_subscribe_port = None
            self.remote_socket = None

        self._is_writer = True
        self._is_local_reader = False
        self.local_reader_rank = -1
        # rank does not matter for remote readers
        self._is_remote_reader = False

        self.handle = Handle(
            connect_ip=connect_ip,
            local_reader_ranks=local_reader_ranks,
            buffer=self.buffer,
            local_subscribe_port=local_subscribe_port,
            remote_subscribe_port=remote_subscribe_port,
        )

    def export_handle(self) -> Handle:
        return self.handle

    @staticmethod
    def create_from_handle(handle: Handle, rank) -> "MessageQueue":
        self = MessageQueue.__new__(MessageQueue)
        self.handle = handle
        self._is_writer = False

        context = Context()

        if rank in handle.local_reader_ranks:
            assert handle.buffer is not None
            self.buffer = handle.buffer
            self.current_idx = 0
            self.local_reader_rank = handle.local_reader_ranks.index(rank)
            self._is_local_reader = True
            self._is_remote_reader = False

            self.local_socket = context.socket(SUB)
            self.local_socket.setsockopt_string(SUBSCRIBE, "")
            self.local_socket.connect(
                f"tcp://{handle.connect_ip}:{handle.local_subscribe_port}"
            )

            self.remote_socket = None
        else:
            self.buffer = None  # type: ignore
            self.current_idx = -1
            self.local_reader_rank = -1
            self._is_local_reader = False
            self._is_remote_reader = True

            self.local_socket = None

            self.remote_socket = context.socket(SUB)
            self.remote_socket.setsockopt_string(SUBSCRIBE, "")
            self.remote_socket.connect(
                f"tcp://{handle.connect_ip}:{handle.remote_subscribe_port}"
            )

        return self

    def wait_until_ready(self):
        """This is a collective operation. All processes (including the
        readers and the writer) should call this function.
        """
        if self._is_writer:
            # wait for all readers to connect
            # local readers
            for i in range(self.n_local_reader):
                # wait for subscription messages from all local readers
                self.local_socket.recv()
            if self.n_local_reader > 0:
                # send a message to all local readers
                # to make sure the publish channel is working
                self.local_socket.send(b"READY")

            # remote readers
            for i in range(self.n_remote_reader):
                # wait for subscription messages from all remote readers
                self.remote_socket.recv()
            if self.n_remote_reader > 0:
                # send a message to all remote readers
                # to make sure the publish channel is working
                self.remote_socket.send(b"READY")
        elif self._is_local_reader:
            # wait for the writer to send a message
            recv = self.local_socket.recv()
            assert recv == b"READY"
        elif self._is_remote_reader:
            # wait for the writer to send a message
            recv = self.remote_socket.recv()
            assert recv == b"READY"

    @contextmanager
    def acquire_write(self):
        assert self._is_writer, "Only writers can acquire write"
        start_time = time.monotonic()
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                read_count = sum(metadata_buffer[1:])
                written_flag = metadata_buffer[0]
                if written_flag and read_count != self.buffer.n_reader:
                    # this block is written and not read by all readers
                    # for writers, `self.current_idx` is the next block to write
                    # if this block is not ready to write,
                    # we need to wait until it is read by all readers
                    # wait for a while
                    time.sleep(RINGBUFFER_SLEEP_INTERVAL)
                    # if we wait for a long time, we should warn the user
                    if (
                        time.monotonic() - start_time
                        > RINGBUFFER_WARNING_INTERVAL * n_warning
                    ):
                        logger.warning(
                            "No available block found in %s second. ",
                            RINGBUFFER_WARNING_INTERVAL,
                        )
                        n_warning += 1
                    continue
                # found a block that is either
                # (1) not written
                # (2) read by all readers

                # mark the block as not written
                metadata_buffer[0] = 0
                # let caller write to the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf
                # caller has written to the buffer
                # NOTE: order is important here
                # first set the read flags to 0
                # then set the written flag to 1
                # otherwise, the readers may think they already read the block
                for i in range(1, self.buffer.n_reader + 1):
                    # set read flag to 0, meaning it is not read yet
                    metadata_buffer[i] = 0
                # mark the block as written
                metadata_buffer[0] = 1
                self.current_idx = (
                    self.current_idx + 1
                ) % self.buffer.max_chunks
                break

    @contextmanager
    def acquire_read(self):
        assert self._is_local_reader, "Only readers can acquire read"
        start_time = time.monotonic()
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                read_flag = metadata_buffer[self.local_reader_rank + 1]
                written_flag = metadata_buffer[0]
                if not written_flag or read_flag:
                    # this block is either
                    # (1) not written
                    # (2) already read by this reader
                    # for readers, `self.current_idx` is the next block to read
                    # if this block is not ready,
                    # we need to wait until it is written
                    # wait for a while
                    time.sleep(RINGBUFFER_SLEEP_INTERVAL)
                    # if we wait for a long time, we should warn the user
                    if (
                        time.monotonic() - start_time
                        > RINGBUFFER_WARNING_INTERVAL * n_warning
                    ):
                        logger.warning(
                            "No available block found in %s second. ",
                            RINGBUFFER_WARNING_INTERVAL,
                        )
                        n_warning += 1
                    continue
                # found a block that is not read by this reader
                # let caller read from the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                # caller has read from the buffer
                # set the read flag
                metadata_buffer[self.local_reader_rank + 1] = 1
                self.current_idx = (
                    self.current_idx + 1
                ) % self.buffer.max_chunks
                break

    def enqueue(self, obj):
        assert self._is_writer, "Only writers can enqueue"
        serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if self.n_local_reader > 0:
            if len(serialized_obj) >= self.buffer.max_chunk_bytes:
                with self.acquire_write() as buf:
                    buf[0] = 1  # overflow
                self.local_socket.send(serialized_obj)
            else:
                with self.acquire_write() as buf:
                    buf[0] = 0  # not overflow
                    buf[1 : len(serialized_obj) + 1] = serialized_obj
        if self.n_remote_reader > 0:
            self.remote_socket.send(serialized_obj)

    def dequeue(self):
        if self._is_local_reader:
            with self.acquire_read() as buf:
                overflow = buf[0] == 1
                if not overflow:
                    # no need to know the size of serialized object
                    # pickle format contains the size information internally
                    # see https://docs.python.org/3/library/pickle.html
                    obj = pickle.loads(buf[1:])
            if overflow:
                recv = self.local_socket.recv()
                obj = pickle.loads(recv)
        elif self._is_remote_reader:
            recv = self.remote_socket.recv()
            obj = pickle.loads(recv)
        else:
            raise RuntimeError("Only readers can dequeue")
        return obj

    def broadcast_object(self, obj=None):
        if self._is_writer:
            self.enqueue(obj)
            return obj
        else:
            return self.dequeue()
