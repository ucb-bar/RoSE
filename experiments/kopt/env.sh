# MB build+run env. source this.
ZCS=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw
source $ZCS/scripts/activate_conda.sh >/dev/null 2>&1
source $ZCS/scripts/set_envvars_sdk.sh >/dev/null 2>&1
export PYTHONPATH=$ZCS
export GLOBAL_CURATED_DIR=$ZCS/modelblaster/kernels
export MB=$ZCS/modelblaster
