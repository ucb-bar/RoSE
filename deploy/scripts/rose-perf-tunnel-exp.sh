#!/bin/bash

ROSE_DIR=$(pwd)

BOOM_CONFIG='firesim-dual-large-boom-fp32gemmini-singlecore-with-airsim-no-nic-l2-llc4mb-ddr3'
ROCKET_CONFIG='firesim-rocket-singlecore-fp32gemmini-with-airsim-fast-no-nic-l2-llc4mb-ddr3'

DNN='trail_resnet14_complex.onnx'
VEL='3.0'
START_Y='-22.85'
END_X='-50.0'
END_CYCLE=40_000_000_000
AIRSIM_STEPS=( 1 2 10 40 )
FIRESIM_CYCLES=( 10_000_000 20_000_000 100_000_000 400_000_000)
LEN_STEPS=${#AIRSIM_STEPS[@]}

ANGLE=200.0

cd ${ROSE_DIR}


# Run Rocket Experiments
echo "RoSE: Updating FireSim Runtime YAML"
yq -i ".target_config.default_hw_config = \"${ROCKET_CONFIG}\"" ${ROSE_DIR}/soc/sim/config/config_runtime_local.yaml
yq -i '.workload.workload_name = "airsim-control-fed.json"' ${ROSE_DIR}/soc/sim/config/config_runtime_local.yaml
bash ${ROSE_DIR}/soc/setup.sh

echo "RoSE: Updating FireMarshal workload YAML"
jq ".command = \"/root/drone_test -m /root/${DNN} -i /root/img_56.png -p unit -x 2 -O 99 -v ${VEL} -l 2\"" ${ROSE_DIR}/soc/sw/rose-images/airsim-control-fed.json > ${ROSE_DIR}/soc/sw/rose-images/tmp.json
mv ${ROSE_DIR}/soc/sw/rose-images/tmp.json ${ROSE_DIR}/soc/sw/rose-images/airsim-control-fed.json 
cd ${ROSE_DIR}/soc/sw

echo "RoSE: Building New Linux Image"
marshal build rose-images/airsim-control-fed.json
cd ${ROSE_DIR}/deploy/hephaestus/


# Run Rocket + Gemmini
echo "Running Rocket+Gemmini"
for (( i=0; i<${LEN_STEPS}; i++ ));
do
    echo "Running RoSE simulation"
    echo ${ANGLE}  > angle.txt
    python3 runner.py -i ${AIRSIM_IP} -r FireSim -a ${AIRSIM_STEPS[$i]} -f ${FIRESIM_CYCLES[$i]} -y ${START_Y} -c ${END_CYCLE} -x ${END_X} -l ${ROSE_DIR}/deploy/hephaestus/logs/rose-perf-tunnel-rocket-gemmini-${FIRESIM_CYCLES[$i]} # | tee ${ROSE_DIR}/deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-${ANGLE_NAMES[$i]}.log
    firesim kill &
    pid=$!
    sleep 10
    kill $pid
    sleep 60
done