#!/usr/bin/python3
# encoding: utf-8
# Runs the "345M" parameter model
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('nnodes', type=int)
parser.add_argument('nproc_per_node', type=int)
parser.add_argument('node_rank', type=int)
parser.add_argument('--master_addr', default="localhost", help="default: %(default)s")
parser.add_argument('--master_port', type=int, default=60000, help="default: %(default)s")
parser.add_argument('--nic', default='mlx5_0')

args = parser.parse_args()

DISTRIBUTED_ARGS=f"--nproc_per_node {args.nproc_per_node} \
--nnodes {args.nnodes} \
--node_rank {args.node_rank} \
--master_addr {args.master_addr} \
--master_port {args.master_port}"

if not args.master_addr in ["localhost", "127.0.0.1"]:
    os.environ['NCCL_IB_HCA'] = f"{args.nic}"
    print('NCCL_IB_HCA', os.environ['NCCL_IB_HCA'])

os.system(f"python -m torch.distributed.launch {DISTRIBUTED_ARGS} \
       torch_distributed_test.py"
)