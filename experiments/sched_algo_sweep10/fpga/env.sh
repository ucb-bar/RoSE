# Sourceable env for the sched_algo_sweep10 FPGA build set.
export XPURT_CODE_ROOT=/scratch2/dima/misc_sw/XPU-RT
export XPURT_DATA_ROOT=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt
# cpsat_scheduler spawns its solve in a subprocess and needs an interpreter
# with ortools; the sweep's AWS venv is not on this host, so one was made here.
export XPURT_CPSAT_PYTHON=/scratch/dima/rose-infra/venv-cpsat/bin/python
export EMIT_PYTHON=/scratch/dima/rose-infra/venv-cpsat/bin/python
