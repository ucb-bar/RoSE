# Octo on-target runs

Logs and comparison tooling for §13 of `../NOTES.md`. **The runs here are from
different configurations** -- several predate fixes made while debugging -- so
read the config column before quoting a number.

| file | target | config | headline |
|---|---|---|---|
| `gold_native.log` + `gold_native_ladder.txt` | native_sim | final: 316 ops, per-channel, 8 real calibration frames, `ACT_PERCENTILE=99.99`, `SPLITFC=1` | `max_abs_err=7`, output cos 0.9756 |
| `gold_spike.log` | spike_riscv64 | same IR as `gold_native` (inspect list stripped so the dumps do not go through HTIF) | the on-target result |
| `native_full_int8.log` | native_sim | 313 ops, **1 noise calibration sample**, no clipping, no SPLITFC | `max_abs_err=26` -- optimistic, see NOTES 13.4 |
| `spike_full_int8.log` | spike_riscv64 | same IR as `native_full_int8` | `max_abs_err=13`, 3,311,894,450 cycles |
| `xtarget_native_nofma.log` | native_sim | as `native_full_int8`, plus a 7-tensor inspect ladder | cross-target evidence |
| `xtarget_native_fma.log` | native_sim | same IR, rebuilt with `EXTRA_KERNEL_CFLAGS='-mfma;-mavx2;-ffp-contract=fast'` | `max_abs_err=20` |
| `xtarget_spike.log` | spike_riscv64 | same IR | `max_abs_err=13` |
| `calibration_spec.json` | -- | the resolved spec the final IR was calibrated with | 8 BridgeData frames + 3 synthetic inputs |

The `gold_native.log` element dumps are elided (1.8 M lines); the numbers are
in `gold_native_ladder.txt`.

## Tooling

```
# device vs PyTorch fp32, per inspected tensor
python inspect_compare.py <run.log> <ir_dir>

# two runs of the SAME IR against each other, per inspected tensor
python xtarget_compare.py <run_a.log> <run_b.log>
```

Both need the IR to have been extracted with `--inspect a,b,c` (see
`EXTRACT_EXTRA_ARGS` in `modelblaster/examples/_run_lib.sh`) and rebuilt
afterwards. Keep the ladder small for a spike run: the dumps go out over
HTIF at roughly 100 lines/s, so 1.8 M lines is hours. Extract with the full
ladder for `RUNNER=native`, then strip `inspect_tensors` from `graph.json`
and rebuild for `RUNNER=spike` -- that is numerically the same graph.
