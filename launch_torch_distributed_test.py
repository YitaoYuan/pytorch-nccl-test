#!/usr/bin/python3
# encoding: utf-8

# Runs the "345M" parameter model
import argparse
import os
import subprocess

def get_pci_device(device_path):
    """Get the PCI device path for a given device."""
    device_link = os.path.join(device_path, 'device')
    if os.path.islink(device_link):
        return os.path.realpath(device_link)
    return None

def get_device_mappings():
    """Get mappings between IB devices and Ethernet devices."""
    ib_to_eth = {}
    eth_to_ib = {}
    ib_devices_path = '/sys/class/infiniband'
    net_devices_path = '/sys/class/net'

    # Traverse all IB devices
    for ib_device in os.listdir(ib_devices_path):
        ib_device_path = os.path.join(ib_devices_path, ib_device)
        ib_pci_device = get_pci_device(ib_device_path)

        if ib_pci_device:
            # Traverse all Ethernet devices
            for eth_device in os.listdir(net_devices_path):
                eth_device_path = os.path.join(net_devices_path, eth_device)
                eth_pci_device = get_pci_device(eth_device_path)

                # If PCI devices match, create a mapping
                if eth_pci_device and eth_pci_device == ib_pci_device:
                    ib_to_eth[ib_device] = eth_device
                    eth_to_ib[eth_device] = ib_device
                    break

    return ib_to_eth, eth_to_ib

def ib_to_eth(ib_device):
    """Convert an IB device name to an Ethernet device name."""
    ib_to_eth_mapping, _ = get_device_mappings()
    return ib_to_eth_mapping.get(ib_device, "Unknown IB device")

def eth_to_ib(eth_device):
    """Convert an Ethernet device name to an IB device name."""
    _, eth_to_ib_mapping = get_device_mappings()
    return eth_to_ib_mapping.get(eth_device, "Unknown Ethernet device")


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
    os.environ['NCCL_SOCKET_IFNAME'] = f"{ib_to_eth(args.nic)}"
    print('NCCL_IB_HCA', os.environ['NCCL_IB_HCA'])
    print('NCCL_SOCKET_IFNAME', os.environ['NCCL_SOCKET_IFNAME'])

cmd = f"torchrun {DISTRIBUTED_ARGS} torch_distributed_test.py"
print(cmd)
os.system(cmd)