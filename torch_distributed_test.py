import os
import sys
import torch
import torch.distributed as dist
import argparse
import time
import random
if os.environ.get("IMPORT_NETCCL") == "1":
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

test_op = ["allreduce", "reduce", "broadcast", "reducescatter", "allgather", "gather", "scatter", "alltoall"]
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
    dist.barrier(device_ids=[torch.cuda.current_device()]) # only do barrier for current device

    partition_numel = numel // world_size
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    barrier_end = torch.cuda.Event(enable_timing=True)
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
    elif op == "barrier":
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        raise NotImplementedError(op)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    end.record()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    barrier_end.record()
    barrier_end.synchronize()
    t = (start.elapsed_time(end) - end.elapsed_time(barrier_end))* 1e-3
    return t

szs = [2**s for s in range(6, 31)]
for op in ["allreduce", "reduce", "broadcast", "reducescatter", "allgather", "barrier"]:
    print(f"test {op}")
    if op == "barrier":
        l = []
        for i in range(7):
            t = op_test(op, 0)
            l += [t]
        l.sort()
        t = l[len(l)//2]
        dist_print(f"time {t:.6f}")
        dist_print(l)
    else:
        for sz in szs:
            l = []
            for i in range(7):
                t = op_test(op, sz)
                l += [t]
            l.sort()
            t = l[len(l)//2]
            bw = sz * 8 * 1e-9 / t
            dist_print(f"size {sz} time {t:.6f} alg_bw {bw:.3f}")
            dist_print(l)
    #print(sbuf[0:256:64])
    #print(sbuf[63:256:64])

print()

# iter = 0
# while True:
#     sz = 128 * 1024 * 1024
#     t = op_test("allreduce", sz)
#     bw = sz * 8 * 1e-9 / t
#     if iter % 10 == 0:
#         dist_print(f"size {sz} time {t:.6f} alg_bw {bw:.3f}")
#     iter += 1

sys.exit(0)

szs = [256 * 1024, 8 * 1024 * 1024, 64 * 1024 * 1024, 1024 * 1024 * 1024]
for op in ["allreduce", "reduce", "broadcast", "reducescatter", "allgather"]:
    print(f"test {op}")
    for sz in szs:
        l = []
        for i in range(3):
            t = op_test(op, sz)
            l += [t]
        bw = sz * 8 * 1e-9 / t
        dist_print(f"size {sz} time {t:.6f} alg_bw {bw:.3f}")
        dist_print(l)
        if op in ["allreduce", "reduce", "broadcast"]:
            print(sbuf[0:128]) # 2 packets
        elif op in ["reducescatter"]:
            print(rbuf[0:128])
        elif op in ["allgather"]:
            for r in range(world_size):
                offset = r * (sz // 4 // world_size)
                print(r, rbuf[offset:offset+128])
    #print(sbuf[0:256:64])
    #print(sbuf[63:256:64])
print()

print("test unaligned broadcast")
unaligned_size = [256 + 3, 256 + 5, 256 + 255]
for sz in unaligned_size:
    byte_sbuf = torch.zeros(512, dtype=torch.uint8).cuda()
    if rank == 0:
        byte_sbuf[:sz].fill_(1)
    dist.broadcast(byte_sbuf[:sz], src=0)
    print(byte_sbuf)
print()

if world_size >= 2:
    print("test multiple groups")

    # generate 2 subgroups
    # test with 3 groups (subgroups + global group)
    # generate random operations
    # check if the result is correct (with eps)

    random.seed(0) # generate the same sequence
    eps = 1e-4
    test_cnt = 1024
    test_size = 1<<20
    ranks = [i for i in range(world_size)]
    groups = [None, dist.new_group(ranks[::2]), dist.new_group(ranks[1::2])]
    test_buf = torch.zeros(test_size//4, dtype=torch.float32).cuda()
    test_buf_init = torch.zeros(test_size//4, dtype=torch.float32).cuda()
    for i in range(len(test_buf_init)):
        test_buf_init[i] = i

    for i in range(test_cnt):
        # if i == 0: # a BUG case (reduce dst=0 or broadcast src=1)
        #     group_id = 0
        #     op_id = 2
        #     random_rank_seed = 1
        # else:
        group_id = random.randint(0, len(groups)-1)
        op_id = random.randint(0, 4) 
        random_rank_seed = random.randint(0, 2**32-1)
        group = groups[group_id]
        if group == dist.GroupMember.NON_GROUP_MEMBER: # this process is not in group
            print(f"group {group_id}, skip")
            continue
        rank = dist.get_rank(group)
        part_len = test_buf_init.numel() // dist.get_world_size(group)
        slicer = slice(part_len*rank, part_len*(rank+1))
        root_local_rank = random_rank_seed % dist.get_world_size(group)
        root_global_rank = root_local_rank if group==None else dist.get_global_rank(group, root_local_rank)
        op = test_op[op_id]
        numel = test_size // 4
        test_buf = test_buf_init.clone()
        need_root = op in ["reduce", "broadcast"]
        print(f"group {group_id}, {op}", end='')
        if need_root:
            print(f", root lrank {root_local_rank}, root grank {root_global_rank}", end='')
        if op == "allreduce":
            dist.all_reduce(test_buf, group=group)
            std_res = test_buf_init * dist.get_world_size(group)
        elif op == "reduce":
            dist.reduce(test_buf, dst=root_global_rank, group=group) # note: reduce/broadcast use global rank as dst/src
            if dist.get_rank(group) == root_local_rank:
                std_res = test_buf_init * dist.get_world_size(group)
            else:
                std_res = test_buf_init
        elif op == "broadcast":
            dist.broadcast(test_buf, src=root_global_rank, group=group)
            std_res = test_buf_init
        elif op == "reducescatter":
            dist.reduce_scatter_tensor(test_buf[slicer], test_buf, group=group)
            std_res = test_buf_init.clone()
            std_res[slicer] = test_buf_init[slicer] * dist.get_world_size(group)
        elif op == "allgather":
            test_buf = torch.zeros_like(test_buf, dtype=torch.float32).cuda()
            test_buf[slicer] = test_buf_init[slicer]
            dist.all_gather_into_tensor(test_buf, test_buf[slicer], group=group)
            std_res = test_buf_init
        
        abs_delta = (test_buf - std_res).abs()
        print(f", max abs delta {abs_delta.max()}", end='')
        rel_delta = (abs_delta / std_res).abs().nan_to_num(nan=float('inf'))
        print(f", max rel delta {rel_delta.max()}", end='')
        min_delta = torch.min(torch.stack([abs_delta, rel_delta]), dim=0)
        max_delta = min_delta.values.max()
        check_result = max_delta < eps

        print(f", result {'OK' if check_result else 'ERROR'}({max_delta:.3g})")
        if not check_result:
            zero_seg = []
            for i in range(len(test_buf)):
                if i > 0 and test_buf[i] < 1:
                    if len(zero_seg) > 0 and zero_seg[-1][1] == i-1:
                        zero_seg[-1][1] = i
                    else:
                        zero_seg.append([i, i])
            print(zero_seg)
            print("result", test_buf)
            print("std result", std_res)
            print("abs delta", abs_delta)
            print("rel delta", rel_delta)
            sys.exit(1)
            
    print()

dist.destroy_process_group()
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

dist.destroy_process_group()