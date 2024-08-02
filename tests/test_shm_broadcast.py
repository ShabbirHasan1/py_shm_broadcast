# Do Checkout https://github.com/vllm-project/vllm #type: ignore
# Thanks to @vllm-project for the awesome project!
from __future__ import annotations

import logging
import multiprocessing
import pickle
import sys
import time
from typing import List

import numpy as np
from shm_broadcast import MessageQueue

logger = logging.getLogger(__name__)


def get_arrays(n: int, seed: int = 0) -> List[np.ndarray]:
    np.random.seed(seed)
    # Each array will have 128 elements
    # with int64, each array will have 1024 bytes or 1kb
    return [np.random.randint(1, 100, 128) for _ in range(n)]


def distributed_run(
    _publisher_fn,
    _consumer_fn,
    megabytes: int = 4,
    no_of_consumers: int = 4,
    seed: int = 64,
):
    processes = []
    _p = multiprocessing.Process(
        target=_publisher_fn,
        args=(
            megabytes,
            no_of_consumers,
            seed,
        ),
    )
    processes.append(_p)
    _p.start()
    time.sleep(5)
    for rank in range(no_of_consumers):
        p = multiprocessing.Process(
            target=_consumer_fn,
            args=(
                megabytes,
                rank,
                seed,
            ),
        )
        processes.append(p)
        p.start()
        time.sleep(5)

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def publisher_fn(megabytes: int, no_of_consumers: int, seed: int):
    N = 1_024 * megabytes
    arrs = get_arrays(N, seed)
    broadcaster = MessageQueue(
        n_reader=no_of_consumers,
        n_local_reader=no_of_consumers,
        local_reader_ranks=list(range(no_of_consumers)),
        max_chunk_bytes=1 * 1024,
        max_chunks=2,
    )
    with open("handle.pickle", "wb") as handle:
        pickle.dump(
            broadcaster.export_handle(),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    broadcaster.wait_until_ready()
    print(
        f"Broadcasting {(1024*N)/1_048_576:.2f}MB (1024 bytes * {N} bytes) of data"
    )
    start = time.perf_counter_ns()
    for x in arrs:
        broadcaster.broadcast_object(x)
    elapsed_ns = time.perf_counter_ns() - start
    (
        elapsed_us,
        elapsed_ms,
        elapsed_s,
    ) = elapsed_ns * 1e-3, elapsed_ns * 1e-6, elapsed_ns * 1e-9
    latency_ns, latency_us, latency_ms, latency_s = (
        elapsed_ns / N,
        elapsed_us / N,
        elapsed_ms / N,
        elapsed_s / N,
    )
    print(
        f"Total time elapsed: {elapsed_ns:.3f} ns, {elapsed_us:.3f} µs, {elapsed_ms:.3f} ms, {elapsed_s:.3f} s"
    )
    print(
        f"Latency: {latency_ns:.3f} ns, {latency_us:.3f} µs, {latency_ms:.3f} ms, {latency_s:.3f} s"
    )


def consumer_fn(megabytes: int, rank: int, seed: int):
    N = 1_024 * megabytes
    arrs = get_arrays(N, seed)
    with open("handle.pickle", "rb") as _handle:
        handle = pickle.load(_handle)
    broadcaster = MessageQueue.create_from_handle(handle, rank=rank)
    broadcaster.wait_until_ready()
    print(
        f"Consumer - {rank:02d} - Receiving {(1024*N)/1_048_576:.2f}MB (1024 bytes * {N} bytes) brodcasted data"
    )
    for x in arrs:
        y = broadcaster.broadcast_object(None)
        assert np.array_equal(x, y)


def test_shm_broadcast(no_of_consumers: int = 4):
    megabytes, no_of_consumers, seed = (
        100,
        4,
        64,
    )
    distributed_run(
        publisher_fn,
        consumer_fn,
        megabytes=megabytes,
        no_of_consumers=no_of_consumers,
        seed=seed,
    )


if __name__ == "__main__":
    megabytes, no_of_consumers, seed = (
        int(sys.argv[1]),
        int(sys.argv[2]),
        int(sys.argv[3]),
    )
    distributed_run(
        publisher_fn,
        consumer_fn,
        megabytes=megabytes,
        no_of_consumers=no_of_consumers,
        seed=seed,
    )
