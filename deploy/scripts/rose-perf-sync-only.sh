#!/bin/bash

cycle_steps=(  500_000_000   200_000_000   100_000_000   50_000_000  20_000_000   10_000_000   5_000_000   2_000_000   1_000_000   500_000   200_000   100_000   50000   20000   10000   5000   2000   1000   )
max_steps=(  50_000_000_000   20_000_000_000   10_000_000_000   5_000_000_000  2_000_000_000   1_000_000_000   500_000_000   200_000_000   100_000_000   50_000_000   20_000_000   10_000_000   5000000   2000000   1000000   500000   200000   100000   )

rm sim_data_test.log
len=${#cycle_steps[@]}
# use for loop read all nameservers
echo "RoSE: Updating FireSim Runtime YAML"
yq -i '.target_config.default_hw_config = "firesim-rocket-singlecore-fp32gemmini-with-airsim-fast-no-nic-l2-llc4mb-ddr3"' ${ROSE_DIR}/soc/sim/config_runtime_local.yaml
yq -i '.workload.workoad_name = "airsim-driver-fed.json"' ${ROSE_DIR}/soc/sim/config_runtime_local.yaml
bash ${ROSE_DIR}/soc/setup.sh

for (( i=0; i<${len}; i++ ));
do
    echo "running sync: ${cycle_steps[$i]} max: ${max_steps[$i]}"
    #echo "running sync: ${cycle_steps[$i]} max: ${cycle_steps[$i]}"
    firesim infrasetup
    python3 runner.py -f ${cycle_steps[$i]} -c ${max_steps[$i]} -r FireSim >> sim_cmd.log
    # python3 runner.py -f ${cycle_steps[$i]} -c ${max_steps[$i]} -r FireSim
    # echo "python3 runner.py -f ${cycle_steps[$i]} -c ${max_steps[$i]} -r MIDAS" >> sim_cmd.log
    # python3 runner.py -f ${cycle_steps[$i]} -c ${max_steps[$i]} -r MIDAS | grep writestring >> sim_data_midas.log
    # python3 runner.py -f ${cycle_steps[$i]} -c ${cycle_steps[$i]}00 -r FireSim
    sleep 60
done
echo "Completed tests!"