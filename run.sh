#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=1,0 NETCCL_BIND_ADDR=192.168.1.1 NETCCL_CONTROLLER_ADDR=10.0.0.100:50051 ./launch_torch_distributed_test.py 2 1 --rdzv-endpoint 10.0.0.1 --nic mlx5_1