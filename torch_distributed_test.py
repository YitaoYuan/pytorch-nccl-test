import torch
import torch.distributed as dist
import argparse
import time
import os

parser = argparse.ArgumentParser()

dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

min_size = 4 * world_size
max_size = (1<<30)

def is_power_of_two(x):
    return x & (x-1) == 0

assert is_power_of_two(min_size) and is_power_of_two(max_size)

test_op = ["allreduce", "allgather", "reducescatter", "reduce", "broadcast"]
test_times = 3
sbuf = torch.zeros(max_size//4, dtype=torch.float32).cuda()
rbuf = torch.zeros(max_size//4, dtype=torch.float32).cuda()
torch.cuda.synchronize()

def dist_print(*args, **kwargs):
    if local_rank == 0:
        print(*args, **kwargs)

def op_test(op, size):
    global sbuf, rbuf
    numel = size // 4
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if op == "allreduce":
        dist.all_reduce(sbuf[:numel])
    elif op == "allgather":
        dist._all_gather_base(rbuf[:numel], sbuf[:numel//world_size])
    elif op == "reducescatter":
        dist._reduce_scatter_base(rbuf[:numel//world_size], sbuf[:numel])
    elif op == "reduce":
        dist.reduce(sbuf[:numel], dst=0)
    elif op == "broadcast":
        dist.broadcast(sbuf[:numel], src=0)
    end.record()
    end.synchronize()
    t = start.elapsed_time(end) * 1e-3
    return t

for op in test_op:
    test_size = min_size
    coeff = 2*(world_size-1)/world_size if op == "allreduce" else (world_size-1)/world_size
    dist_print(f"{op}:")

    for _ in range(100): # warm up
        op_test(op, test_size)

    while test_size <= max_size:
        test_size_gbits = test_size*8*1e-9
        times = []
        for i in range(test_times):
            t = op_test(op, test_size)
            times.append(t)
        
        avg_t = sum(times)/len(times)
        alg_bw = test_size_gbits / avg_t
        bus_bw = coeff * alg_bw

        dist_print(f"size {test_size} time {t:.6f} alg_bw {alg_bw:.3f} bus_bw {bus_bw:.3f}")
        test_size *= 2
