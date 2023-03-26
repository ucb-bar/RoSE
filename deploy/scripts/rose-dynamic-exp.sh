#!/bin/bash

PROJECT_ROOT="/scratch/$(whoami)"
ROSE_DIR=${PROJECT_ROOT}/RoSE

BOOM_CONFIG='firesim-dual-large-boom-fp32gemmini-singlecore-with-airsim-no-nic-l2-llc4mb-ddr3'
ROCKET_CONFIG='firesim-rocket-singlecore-fp32gemmini-with-airsim-fast-no-nic-l2-llc4mb-ddr3'

START_Y='0.0'
END_X='-80.0'
END_CYCLE=40_000_000_000
AIRSIM_STEPS=2
FIRESIM_CYCLES=20_000_000
ANGLE=200.0
DNN='trail_resnet14_complex.onnx'
DNN_SMALL='trail_resnet6_complex.onnx'
VELS=9

cd ${ROSE_DIR}



# Run BOOM Experiments
echo "RoSE: Updating FireSim Runtime YAML"
yq -i ".target_config.default_hw_config = \"${BOOM_CONFIG}\"" ${ROSE_DIR}/soc/sim/config_runtime_local.yaml
yq -i '.workload.workoad_name = "airsim-control-fed.json"' ${ROSE_DIR}/soc/sim/config_runtime_local.yaml
bash ${ROSE_DIR}/soc/setup.sh

# Run BOOM + Gemmini
echo "Running BOOM+Gemmini"
echo "RoSE: Updating FireMarshal Workload YAML"
jq ".command = \"/root/drone_dynamic_test -m /root/${DNN} -m /root/${DNN_SMALL} -i /root/img_56.png -p unit -x 2 -O 99 -v ${VELS[$i]} -w 10.0 \"" ${ROSE_DIR}/soc/sw/rose-images/airsim-control-fed.json > ${ROSE_DIR}/soc/sw/rose-images/tmp.json
mv ${ROSE_DIR}/soc/sw/rose-images/tmp.json ${ROSE_DIR}/soc/sw/rose-images/airsim-control-fed.json 
cd ${ROSE_DIR}/soc/sw

echo "RoSE: Building New Linux Image"
marshal build rose-images/airsim-control-fed.json
cd ${ROSE_DIR}/deploy/hephaestus/

echo "Running RoSE simulation"
echo ${ANGLE}  > angle.txt
python3 runner.py -r FireSim -a ${AIRSIM_STEPS} -f ${FIRESIM_CYCLES} -y ${START_Y} -c ${END_CYCLE} -x ${END_X} -l ${ROSE_DIR}/deploy/hephaestus/logs/rose-dynamic-exp-boom-gemmini # | tee ${ROSE_DIR}/deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-${ANGLE_NAMES[$i]}.log
firesim kill &
pid=$!
sleep 10
kill $pid
sleep 60