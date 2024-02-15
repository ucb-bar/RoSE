#!/usr/bin/env python3
# A script for building the target packettest application

import os
import argparse

ROSE_SW_DIR = os.path.dirname(os.path.realpath(__file__))
ROSE_HEADER_DIR = os.path.join(ROSE_SW_DIR, "generated-src", "rose_c_header")
ROSE_PORT_HEADER = os.path.join(ROSE_HEADER_DIR, "rose_port.h")
ROSE_PACKET_HEADER = os.path.join(ROSE_HEADER_DIR, "rose_packet.h")

PACKETTEST_SRC_DIR = os.path.join(ROSE_SW_DIR, "rose-images", "airsim-packettest")
CY_TEST_DIR = os.path.join(ROSE_SW_DIR, "..", "sim", "firesim", "target-design", "chipyard", "tests")
FSIM_WORKLOAD_DIR = os.path.join(ROSE_SW_DIR, "..", "sim", "firesim", "deploy", "workloads", "airsim-packettest")

# argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help='The target C file to build', default='all')
    args = parser.parse_args()
   
    # copying the sources
    os.system(f"cp -f {ROSE_PORT_HEADER} {CY_TEST_DIR}")
    os.system(f"cp -f {ROSE_PACKET_HEADER} {CY_TEST_DIR}")

    if args.target != 'all':
        # make sure the target is in src_dir
        target_src = os.path.join(PACKETTEST_SRC_DIR, f"{args.target}.c")
        if not os.path.exists(target_src):
            print(f"Error: {target_src} does not exist")
            exit(1)
        os.system(f"cp -f {target_src} {CY_TEST_DIR}")
    
    else:
        # iterate over the source directory
        for file in os.listdir(PACKETTEST_SRC_DIR):
            if file.endswith(".c"):
                os.system(f"cp -f {os.path.join(PACKETTEST_SRC_DIR, file)} {CY_TEST_DIR}")

        # build the packettest application
        os.chdir(CY_TEST_DIR)
        os.system("make clean")

        if args.target != 'all':
            os.system(f"export $PROGRAMS={args.target}")
            os.system("make")
            os.system(f"cp -f {args.target}.riscv {FSIM_WORKLOAD_DIR}")
            
        else:
            for file in os.listdir(CY_TEST_DIR):
                if file.endswith(".c"):
                    os.system(f"export $PROGRAMS={file[:-2]}")
                    os.system("make")
                    os.system(f"cp -f {file[:-2]}.riscv {FSIM_WORKLOAD_DIR}")