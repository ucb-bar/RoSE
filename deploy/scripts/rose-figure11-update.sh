#!/bin/bash

ROSE_DIR=$(pwd)

BOOM_CONFIG='firesim-dual-large-boom-fp32gemmini-singlecore-with-airsim-no-nic-l2-llc4mb-ddr3'
ROCKET_CONFIG='firesim-rocket-singlecore-fp32gemmini-with-airsim-fast-no-nic-l2-llc4mb-ddr3'

VEL='9.0'
START_Y='0.0'
END_X='-80.0'
END_CYCLE=55_000_000_000
AIRSIM_STEPS=2
FIRESIM_CYCLES=20_000_000
ANGLE=200.0

DNNS=( 'resnet34' )
LEN_DNN=${#DNNS[@]}

cd ${ROSE_DIR}


# Run BOOM Experiments
echo "RoSE: Updating FireSim Runtime YAML"
yq -i ".target_config.default_hw_config = \"${BOOM_CONFIG}\"" ${ROSE_DIR}/soc/sim/config/config_runtime_local.yaml
yq -i '.workload.workload_name = "airsim-control-fed.json"' ${ROSE_DIR}/soc/sim/config/config_runtime_local.yaml
bash ${ROSE_DIR}/soc/setup.sh

# Run BOOM + Gemmini
echo "Running BOOM+Gemmini"
for (( i=0; i<${LEN_DNN}; i++ ));
do
    echo "RoSE: Updating FireMarshal Workload YAML"
    jq ".command = \"/root/drone_test -m /root/trail_${DNNS[$i]}_complex.onnx -i /root/img_56.png -p unit -x 2 -O 99 -v ${VEL}\"" ${ROSE_DIR}/soc/sw/rose-images/airsim-control-fed.json > ${ROSE_DIR}/soc/sw/rose-images/tmp.json
    mv ${ROSE_DIR}/soc/sw/rose-images/tmp.json ${ROSE_DIR}/soc/sw/rose-images/airsim-control-fed.json 
    cd ${ROSE_DIR}/soc/sw

    echo "RoSE: Building New Linux Image"
    marshal clean rose-images/airsim-control-fed.json
    marshal build rose-images/airsim-control-fed.json
    cd ${ROSE_DIR}/deploy/hephaestus/


    echo "Running RoSE simulation"
    echo ${ANGLE}  > angle.txt
    python3 runner.py -i ${AIRSIM_IP} -r FireSim -a ${AIRSIM_STEPS} -f ${FIRESIM_CYCLES} -y ${START_Y} -c ${END_CYCLE} -x ${END_X} -l ${ROSE_DIR}/deploy/hephaestus/logs/rose-hw-sw-sweep-boom-gemmini-${DNNS[$i]} # | tee ${ROSE_DIR}/deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-${ANGLE_NAMES[$i]}.log
    firesim kill &
    pid=$!
    sleep 10
    kill $pid
    sleep 60
done
