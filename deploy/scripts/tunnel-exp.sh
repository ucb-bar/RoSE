#!/bin/bash
PROJECT_ROOT="/scratch/$(whoami)"
ROSE_DIR=${PROJECT_ROOT}/RoSE

BOOM_CONFIG='firesim-dual-large-boom-fp32gemmini-singlecore-with-airsim-no-nic-l2-llc4mb-ddr3'
ROCKET_CONFIG='firesim-rocket-singlecore-fp32gemmini-with-airsim-fast-no-nic-l2-llc4mb-ddr3'

DNN='trail_resnet14_complex.onnx'
VEL='3.0'
START_Y='-22.7'
END_X='-50.0'
END_CYCLE=40_000_000_000
AIRSIM_STEPS=2
FIRESIM_CYCLES=20_000_000

ANGLES=( '180.0' '200.0' '160.0 ')
ANGLE_NAMES=( '180' '200' '160' )
LEN_ANGLE=${#ANGLES[@]}

cd ${ROSE_DIR}


# Run Rocket Experiments
echo "RoSE: Updating FireSim Runtime YAML"
yq -i ".target_config.default_hw_config = \"${ROCKET_CONFIG}\"" ${ROSE_DIR}/soc/sim/config_runtime_local.yaml
yq -i '.workload.workload_name = "airsim-control-fed.json"' ${ROSE_DIR}/soc/sim/config_runtime_local.yaml
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
for (( i=0; i<${LEN_ANGLE}; i++ ));
do
    echo "Running RoSE simulation"
    echo ${ANGLES[$i]}  > angle.txt
    python3 runner.py -r FireSim -a ${AIRSIM_STEPS} -f ${FIRESIM_CYCLES} -y ${START_Y} -c ${END_CYCLE} -x ${END_X} -l ${ROSE_DIR}/deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-${ANGLE_NAMES[$i]} # | tee ${ROSE_DIR}/deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-${ANGLE_NAMES[$i]}.log
    firesim kill &
    pid=$!
    sleep 10
    kill $pid
    sleep 60
done

# Run BOOM Experiments
echo "RoSE: Updating FireSim Runtime YAML"
yq -i ".target_config.default_hw_config = \"${BOOM_CONFIG}\"" ${ROSE_DIR}/soc/sim/config_runtime_local.yaml
yq -i '.workload.workload_name = "airsim-control-fed.json"' ${ROSE_DIR}/soc/sim/config_runtime_local.yaml
bash ${ROSE_DIR}/soc/setup.sh

# Run BOOM + Gemmini
echo "Running Boom+Gemmini"
for (( i=0; i<${LEN_ANGLE}; i++ ));
do
    echo "Running RoSE simulation"
    echo ${ANGLES[$i]}  > angle.txt
    python3 runner.py -r FireSim -a ${AIRSIM_STEPS} -f ${FIRESIM_CYCLES} -y ${START_Y} -c ${END_CYCLE} -x ${END_X} -l ${ROSE_DIR}/deploy/hephaestus/logs/tunnel-exp-boom-gemmini-${ANGLE_NAMES[$i]} # | tee ${ROSE_DIR}/deploy/hephaestus/logs/tunnel-exp-boom-gemmini-${ANGLE_NAMES[$i]}.log
    firesim kill &
    pid=$!
    sleep 10
    kill $pid
    sleep 60
done

echo "RoSE: Updating FireMarshal workload YAML"
jq ".command = \"/root/drone_test -m /root/${DNN} -i /root/img_56.png -p unit -x 0 -O 99 -v ${VEL} -l 2\"" ${ROSE_DIR}/soc/sw/rose-images/airsim-control-fed.json > ${ROSE_DIR}/soc/sw/rose-images/tmp.json
mv ${ROSE_DIR}/soc/sw/rose-images/tmp.json ${ROSE_DIR}/soc/sw/rose-images/airsim-control-fed.json 
cd ${ROSE_DIR}/soc/sw

echo "RoSE: Building New Linux Image"
marshal build rose-images/airsim-control-fed.json
cd ${ROSE_DIR}/deploy/hephaestus/

# Run BOOM Only
echo "Running Boom Only"
for (( i=0; i<${LEN_ANGLE}; i++ ));
do
    echo "Running RoSE simulation"
    echo ${ANGLES[$i]}  > angle.txt
    python3 runner.py -r FireSim -a ${AIRSIM_STEPS} -f ${FIRESIM_CYCLES} -y ${START_Y} -c ${END_CYCLE} -x ${END_X} -l ${ROSE_DIR}/deploy/hephaestus/logs/tunnel-exp-boom-only-${ANGLE_NAMES[$i]} # | tee ${ROSE_DIR}/deploy/hephaestus/logs/tunnel-exp-boom-only-${ANGLE_NAMES[$i]}.log
    firesim kill &
    pid=$!
    sleep 10
    kill $pid
    sleep 60
done