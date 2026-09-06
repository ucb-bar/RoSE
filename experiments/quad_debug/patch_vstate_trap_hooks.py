#!/usr/bin/env python3
"""Binary-patch the isr.S trap-entry/exit V hooks out of an xpurt ELF.

WHY THIS EXISTS.  The quad-only faults are the trap-path vector corruption
already root-caused in modelblaster commit 5057b0e: arch/riscv/core/isr.S
calls z_riscv_vstate_save on EVERY trap entry and z_riscv_vstate_restore on
every trap exit, and that call issues `csrw vstart,0` + `vsetvli e8,m8` +
4x vse8.v into Saturn's DECOUPLED vector unit while the interrupted thread's
own vector memop may still be in flight.  On this SoC that corrupts the
interrupted thread's registers.

This script tests that causally on the exact faulting binaries, with no
rebuild (the shared modelblaster build tree is in use by another campaign):
it writes c.ret (0x8082) at the entry of BOTH isr.S-side wrappers.

It deliberately does NOT touch z_riscv_vstate_save_thread /
z_riscv_vstate_restore_thread, which switch.S calls directly -- so per-thread
V context switching keeps working and numerics stay verifiable.  That is safe
here because CONFIG_RISCV_V_KERNEL_ONLY=y strips V from the global -march, so
no ISR can emit a vector instruction and clobber v0..v31 behind the
interrupted thread's back.

The script asserts each wrapper has exactly ONE caller before patching; if
that ever stops being true the assumption above needs re-checking.

  patch_vstate_trap_hooks.py <in.elf> <out.elf>
"""
import subprocess
import sys
import os

NM = "riscv64-zephyr-elf-nm"
OBJDUMP = "riscv64-zephyr-elf-objdump"
READELF = "riscv64-zephyr-elf-readelf"
TARGETS = ("z_riscv_vstate_save", "z_riscv_vstate_restore")
C_RET = b"\x82\x80"          # c.ret, little endian


def run(*a):
    return subprocess.run(a, capture_output=True, text=True, check=True).stdout


def symbols(elf):
    out = {}
    for line in run(NM, elf).splitlines():
        p = line.split()
        if len(p) == 3:
            out[p[2]] = int(p[0], 16)
    return out


def vaddr_to_off(elf, va):
    """Map a virtual address to a file offset using the program headers."""
    for line in run(READELF, "-lW", elf).splitlines():
        p = line.split()
        if len(p) >= 6 and p[0] == "LOAD":
            off = int(p[1], 16)
            vad = int(p[2], 16)
            fsz = int(p[4], 16)
            if vad <= va < vad + fsz:
                return off + (va - vad)
    raise SystemExit(f"vaddr 0x{va:x} not in any LOAD segment")


def callers(elf, name, va):
    """Count call sites resolved by objdump's `# <addr> <name>` comments."""
    dis = run(OBJDUMP, "-d", elf)
    needle = f"{va:x} <{name}>"
    n = 0
    for line in dis.splitlines():
        if needle in line and ("jalr" in line or "\tjr\t" in line or "jal" in line):
            n += 1
    return n


def main():
    src, dst = sys.argv[1], sys.argv[2]
    syms = symbols(src)
    data = bytearray(open(src, "rb").read())
    for t in TARGETS:
        if t not in syms:
            raise SystemExit(f"{src}: no symbol {t}")
        va = syms[t]
        nc = callers(src, t, va)
        if nc != 1:
            raise SystemExit(
                f"{src}: {t} has {nc} call sites, expected exactly 1 "
                f"(the isr.S wrapper); refusing to patch")
        off = vaddr_to_off(src, va)
        print(f"  {t}: va=0x{va:x} off=0x{off:x} "
              f"was={data[off:off+2].hex()} -> c.ret  (callers={nc})")
        data[off:off + 2] = C_RET
    open(dst, "wb").write(bytes(data))
    os.chmod(dst, 0o755)
    print(f"  wrote {dst} ({len(data)} bytes)")


main()
