#!/bin/bash

PROJECT_ROOT="/scratch/$(whoami)"
ROSE_DIR=${PROJECT_ROOT}/RoSE

cycle_steps=(  500_000_000   100_000_000   50_000_000  20_000_000   10_000_000   5_000_000   1_000_000   500_000   100_000   50000   10000   5000   1000   )
max_steps=(  50_000_000_000   10_000_000_000   5_000_000_000  2_000_000_000   1_000_000_000   500_000_000   100_000_000   50_000_000   10_000_000   5000000   1000000   500000   100000   )

rm sim_data_test.log
len=${#cycle_steps[@]}
# use for loop read all nameservers
echo "RoSE: Updating FireSim Runtime YAML"
yq -i '.target_config.default_hw_config = "firesim-rocket-singlecore-fp32gemmini-with-airsim-fast-no-nic-l2-llc4mb-ddr3"' ${ROSE_DIR}/soc/sim/config_runtime_local.yaml
yq -i '.workload.workload_name = "airsim-driver-fed.json"' ${ROSE_DIR}/soc/sim/config_runtime_local.yaml
bash ${ROSE_DIR}/soc/setup.sh

cd ${ROSE_DIR}/soc/sw/
echo "RoSE: Building New Linux Image"
marshal build rose-images/airsim-driver-fed.json
cd ${ROSE_DIR}/deploy/hephaestus/


cd ${ROSE_DIR}/deploy/hephaestus/
rm -rf sim_data_test.log
for (( i=0; i<${len}; i++ ));
do
    echo "running sync: ${cycle_steps[$i]} max: ${max_steps[$i]}"
    #echo "running sync: ${cycle_steps[$i]} max: ${cycle_steps[$i]}"
    python3 runner.py -f ${cycle_steps[$i]} -c ${max_steps[$i]} -r FireSim -l ${ROSE_DIR}/deploy/hephaestus/logs/rose-perf-sync-only-rocket-gemmini-${cycle_steps[$i]}
    # python3 runner.py -f ${cycle_steps[$i]} -c ${max_steps[$i]} -r FireSim
    # echo "python3 runner.py -f ${cycle_steps[$i]} -c ${max_steps[$i]} -r MIDAS" >> sim_cmd.log
    # python3 runner.py -f ${cycle_steps[$i]} -c ${max_steps[$i]} -r MIDAS | grep writestring >> sim_data_midas.log
    # python3 runner.py -f ${cycle_steps[$i]} -c ${cycle_steps[$i]}00 -r FireSim
    firesim kill &
    pid=$!
    sleep 10
    kill $pid
    sleep 60
done
echo "Elapsed Time (sec), Throughput (cycles/sec), Cycle Limit, FireSim Step, AirSim Step" > logs/rose-perf-sync-only.csv
cat sim_data_test.log >> logs/rose-perf-sync-only.csv
echo "Completed tests!"