# third-party patches

`third-party/{executorch,XNNPACK}` are gitlinks in `zephyr-chipyard-sw`, so git
does not track edits made inside them — a fresh `git submodule update` silently
throws these away, and the build then fails or (worse) measures the wrong thing.
They are exported here so they survive.

`third_party_rvv_hetero.patch` — three changes, all required to run ExecuTorch on
a HETEROGENEOUS RISC-V bitstream whose boot hart has no vector unit
(`f2_quad_hetero_norose_tacit_q31_60mhz`). See FINDINGS.md §7.

1. `XNNPACK/CMakeLists.txt` — compile `src/configs/hardware-config.c` with
   `-march=rv64gcv`. It contains a raw `vsetvli` but is not a micro-kernel
   source, so it never got the per-file V flag; under RoSE's
   `CONFIG_RISCV_V_KERNEL_ONLY` (which strips `v` from the global `-march`) it
   fails to ASSEMBLE.
2. `XNNPACK/src/configs/hardware-config.c` — `XNN_RISCV_VLENB` build-time VLEN in
   place of the runtime `vsetvli` probe. ExecuTorch constructs its XNNPACK
   backend in a C++ static initializer, so that probe runs before `main()` on
   the boot hart — hart 0, no vector unit — and traps. Not fixable by pinning.
3. `executorch/.../profiling/XNNProfiler.cpp` — emit the per-op `>>` lines with
   `printf` rather than `ET_LOG(Info)`, which is compiled out at the sample's
   usual `EXECUTORCH_LOG_LEVEL=Error` (a profiling build then silently produces
   no profile at all).

Apply from `zephyr-chipyard-sw/`:

    git apply -p1 --directory=. experiments/.../third_party_rvv_hetero.patch
    # or: patch -p1 -d . < third_party_rvv_hetero.patch
