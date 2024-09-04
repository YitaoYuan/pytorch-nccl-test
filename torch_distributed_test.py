import torch
import torch.distributed as dist
import argparse
import time
import os
import netccl

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

# assert is_power_of_two(min_size) and is_power_of_two(max_size)

test_op = ["allreduce", "allgather", "reducescatter", "reduce", "broadcast", "gather", "scatter", "alltoall"]
warmup_times = 1
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
    sbuf[:numel].fill_(1)
    rbuf[:numel].fill_(-1)
    if op == "broadcast" and rank != 0:
        sbuf[:numel].fill_(-1)

    torch.cuda.synchronize()
    dist.barrier()

    partition_numel = numel // world_size
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if op == "allreduce":
        dist.all_reduce(sbuf[:numel])
        # work = dist.all_reduce(sbuf[:numel], async_op=True)
        # work.wait()
    elif op == "allgather":
        dist.all_gather_into_tensor(rbuf[:numel], sbuf[:partition_numel])
    elif op == "reducescatter":
        dist.reduce_scatter_tensor(rbuf[:partition_numel], sbuf[:numel])
    elif op == "reduce":
        dist.reduce(sbuf[:numel], dst=0) # NOTE: rank 0 finishes later in reduce
    elif op == "broadcast":
        dist.broadcast(sbuf[:numel], src=0) # NOTE: rank 0 finishes earlier in broadcast
    elif op == "gather": # no _gather_base in pytorch
        dst = 0
        gather_list = [rbuf[partition_numel * i : partition_numel * (i+1)] for i in range(world_size)] if rank == dst else None
        dist.gather(sbuf[:partition_numel], gather_list, dst=dst)
    elif op == "scatter": # no _scatter_base in pytorch
        src = 0
        scatter_list = [sbuf[partition_numel * i : partition_numel * (i+1)] for i in range(world_size)] if rank == src else None
        dist.scatter(rbuf[:partition_numel], scatter_list, src=src)
    elif op == "alltoall":
        dist.all_to_all_single(rbuf[:numel], sbuf[:numel])
    end.record()
    end.synchronize()
    t = start.elapsed_time(end) * 1e-3
    return t

szs = [256 * 1024, 8 * 1024 * 1024, 64 * 1024 * 1024, 1024 * 1024 * 1024]
for op in ["allreduce", "reduce", "broadcast"]:
    print(op)
    for sz in szs:
        l = []
        for i in range(3):
            t = op_test(op, sz)
            l += [t]
        bw = sz * 8 * 1e-9 / t
        dist_print(f"size {sz} time {t:.6f} alg_bw {bw:.3f}")
        dist_print(l)
    print(sbuf[0:256:64])
    print(sbuf[63:256:64])
exit(0)

for op in test_op:
    test_size = min_size

    # coeff: refer to https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
    if op == "allreduce":
        coeff = 2*(world_size-1)/world_size
    elif op in ["broadcast", "reduce"]:
        coeff = 1
    else:
        coeff = (world_size-1)/world_size

    dist_print(f"{op}:")

    try: # some op may not be supported by the old version pytorch
        op_test(op, test_size)
    except Exception as e:
        print(e)
        continue

    for _ in range(100): # per-op warm up
        op_test(op, test_size)

    while test_size <= max_size:
        test_size_gbits = test_size*8*1e-9
        times = []
        for i in range(warmup_times + test_times):
            t = op_test(op, test_size)
            if i >= warmup_times: # skip per-size warm up
                times.append(t)

        avg_t = sum(times)/len(times)
        alg_bw = test_size_gbits / avg_t
        bus_bw = coeff * alg_bw

        dist_print(f"size {test_size} time {avg_t:.6f} alg_bw {alg_bw:.3f} bus_bw {bus_bw:.3f}")
        dist_print(times)
        test_size *= 2