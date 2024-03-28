#!/usr/bin/python3
# encoding: utf-8

# Runs the "345M" parameter model
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('nnodes', type=int)
parser.add_argument('nproc_per_node', type=int)
parser.add_argument('--rdzv-endpoint', default="localhost:0", help="default: %(default)s")
parser.add_argument('--nic', default='mlx5_0')

args = parser.parse_args()

# --rdzv-backend: "c10d" (recommanded). for static rendezvous, use "static"
# --rdzv-endpoint: like --master-addr, but --master-addr is for static rendezvous, and --rdzv-endpoint is for dynamic
# --rdzv-id: like a global job ID, should be set the same on all nodes
DISTRIBUTED_ARGS=f"--nnodes {args.nnodes} \
--nproc-per-node {args.nproc_per_node} \
--rdzv-backend c10d \
--rdzv-endpoint {args.rdzv_endpoint} \
--rdzv-id 0 \
"

if not args.rdzv_endpoint.startswith("localhost") and not args.rdzv_endpoint.startswith("127.0.0.1"):
    os.environ['NCCL_IB_HCA'] = f"{args.nic}"
    print('NCCL_IB_HCA', os.environ['NCCL_IB_HCA'])

cmd = f"torchrun {DISTRIBUTED_ARGS} torch_distributed_test.py"
print(cmd)
os.system(cmd)