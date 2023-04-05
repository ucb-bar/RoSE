# RoSÉ: A Hardware-Software Co-Simulation Infrastructure Enabling Pre-Silicon Full-Stack Robotics SoC Evaluation

Robotic systems such as unmanned autonomous drones and self-driving cars have been widely deployed in many scenarios and have the potential to revolutionize the future generation of computing. To improve the performance and energy efficiency of robotic platforms,  significant research efforts are being devoted to developing hardware accelerators for workloads that form bottlenecks in the robotics software pipeline.

Although domain-specific accelerators (DSAs) can offer improved efficiency over general-purpose processors on isolated robotics benchmarks, system-level constraints such as data movement and contention over shared resources could significantly impact the achievable acceleration in an end-to-end fashion.

In addition, the closed-loop nature of robotic systems, where there is a tight interaction across different deployed environments, software stacks, and hardware architecture, further exacerbates the difficulties of evaluating robotic SoCs. 

To address this limitation, we develop RoSÉ, an open-source, hardware-software co-simulation infrastructure for full-stack, pre-silicon hardware-in-the-loop evaluation of robotics SoCs, together with the full software stack and realistic robotic environments created to support robotics RoSE captures the complex interactions across hardware, algorithm, and environment, enabling new architectural research directions in hardware-software co-design for robotic UAV systems.

Specifically, RoSÉ integrates AirSim, a software robotic environment simulator, supporting UAV models and FireSim, an FPGA-based RTL simulator, to capture closed-loop interactions between environments, algorithms, and hardware. The sensor environment data is passed from AirSim to FireSim to trigger the hardware pipeline, and the computed commands are generated from the FireSim hardware simulation and fed back to AirSim. RoSÉ accurately synchronizes the timing and data transfer between the AirSim environment simulator and the FireSim hardware simulator without significant degradation in simulation speed. 

Our evaluation demonstrates that RoSÉ holistically captures the closed-loop interactions between environment, algorithms, and hardware, opening up new opportunities for systematic hardware-software co-design for robotic UAV systems. 

In summary, this project makes the following contributions:

* We build RoSÉ, a hardware-software co-simulation infrastructure for pre-silicon, full-stack evaluation of robotics UAV SoCs. RoSÉ captures the dynamic interactions between robotic hardware, software, and environment\footnote{We commit to an open-source release of the RoSÉ framework.

* We design software algorithms, and hardware SoCs along with robotic environments, to holistically evaluate the co-design trade-off in robotic UAV systems using RoSÉ.

* We demonstrate design space exploration of domain-specific SoCs enabled by RoSÉ and identify significant design trade-offs.

In summary, RoSÉ enables robotics UAV researchers and architects to comprehensively evaluate robotics UAV SoCs. RoSÉ captures a full-stack simulation of a robot by integrating the simulation of a robot UAV's environment and its RTL, enabling design space exploration of robot environments, algorithms, hardware, and system parameters within a unified simulation environment. 

By using RoSÉ, researchers can better study and analyze the tradeoffs of robotic UAV systems without the overhead of taping out an SoC. Building upon RoSÉ by introducing new applications, robot UAV environments, and architectures will enable the agile development of robotics SoCs across diverse domains. 

# RoSÉ Tutorial
This artifact appendix describes how to use RoSÉ to run end-to-end robotics simulations, and how to reproduce experimental results.

The instructions assume that a user already has robotic applications and hardware configurations developed, and provides reference examples used in the evaluation of this work. While RoSÉ can be used to develop new hardware and software, instructions to do so are outside of the scope of the artifact evaluation.

## Tutorial Meta-Information Checklist

* Runtime environment: Ubuntu 18.04.6 LTS, Vitis v2021.1
* Hardware (FireSim Simulation):  Intel Xeon Gold 6242, Xilinx U250
* Hardware (AirSim Simulation): AWS EC2 Instance (g4dn.2xlarge), Intel Xeon Platinum 8259CL, Tesla T4.
* How much disk space is required?: 200GB
* Experiments: AirSim/FireSim end-to-end full stack simulations of a UAV using RoSE, running DNN-based controllers. Experiments evaluate both UAV and simulator performance.
* Program: Chisel (RTL), C++ (FireSim bridge drivers, robotic control software), Python (Synchronizer and scheduler.)
* Quantitative Metrics: DNN Latency, mission time, average flight velocity, accelerator activity factor.
* Qualitative Metrics: Flight trajectories, flight recordings.
* Output: CSV logs from the synchronizer, tracking UAV dynamics, sensing requests, and control targets.
* How much time is needed to prepare the workflow?: 2 hours (scripted installation).
* How much time is needed to complete experiments?: 24 hours (scripted run, scripted result parsing)
* Publicly available: Yes.
* Code licenses: Several, see download.
* Contact for Artifact Evaluator: Contact SLICE support (slice-support@eecs.berkeley.edu) if you need help setting up AWS instances.

## Description
(1) How to access:

 • The artifacts consist of modifications/patches to the following sources:

* RoSÉ Core: Deployment, synchronization, and evaluation software, as well as hardware configurations, and patches to FireSim and Chipyard. (https://github.com/CobbledSteel/RoSE)
* FireSim: Top-level FPGA-Accelerated RTL Simulation Environment (https://github.com/firesim/firesim) 
* Chipyard: RISC-V SoC generation environment (https://github.com/ucb-bar/chipyard) 
* RISCV ONNX Runtime: Software for executing HW-accelerated DNN models, modified for use in  UAV control (https://github.com/ucb-bar/onnxruntime-riscv/tree/onnx-rose).

Additionally, this evaluation builds upon the following infrastructures. For the purpose of the evaluation, binaries for simulators built from Unreal Engine and AirSim are provided. 

* Unreal Engine: 3D Environment development, simulation, and rendering platform (https://www.unrealengine.com/en-US/ue-on-github)
* AirSim: UAV simulation plugin for Unreal Engine (https://github.com/microsoft/AirSim)

(2) Dependencies - Hardware 

To run a full simulation with RoSÉ, access to a GPU and FPGA is required, although these can be hosted on separate computers. For this artifact evaluation, instructions for running simulations on a locally-provisioned FPGA are provided. However, RoSEcan also be used using AWS EC2 FPGA instances (e.g. f1.2xlarge). To avoid lengthy build times of over 8 hours, we provide pre-built FPGA instances for the SoC configurations evaluated in this work. However, reference build scripts for re-generating new bitstreams are provided. 

Additionally, GPU access is needed in order to run robotics environment simulations with rendering. For this evaluation, AirSim binaries packaged using Unreal Engine are provided. We also provide API access to running AirSim environments hosted on AWS. While direct access to live AirSim renderings are not needed for the evaluation, remote desktop access could be provided if requested to view simulation progress.

• One AWS EC2 c5.4xlarge instance (also referred to as “manager” instance), and three f1.2xlarge instances are required (we split the workload to run on three parallel f1 instances to save runtime, which takes more than 8 hours if run on a single instance). The latter will be launched automatically by FireSim’s manager.

• We have provided pre-built FPGA images to avoid the long latency (sim 8 hours) of the FPGA-built process. However, if users want to build custom FPGA images, one additional z1d.2xlarge is required.

(3) Dependencies - Software 

Use ssh or mosh on your local machine to remote access evaluation instances. All other requirements are automatically installed by scripts in the following sections. 

## Installation

Evaluation is performed on EDA servers provided by the authors. To access the evaluation instances, provide public keys for remote login, and the authors will provide server addresses for evaluation. Within the evaluation instances, run all `installation/setup` within the `/scratch/rose-isca-ae-[reviewer-login-id]` directory, referred to as the project directory. An alternative way of setting  up is to refer to FireSim docs for on premise FPGA setup https://docs.fires.im/en/1.14.2/Advanced-Usage/Vitis.html?highlight=on-premise The evaluation instances will have the environment set up for accessing the necessary EDA tools.

Running all the steps below in a screen or tmux session is recommended, as some commands may take several hours to execute.

• FireSim Installation Begin by installing FireSim on the evaluation instances by running the following commands.

```
    git clone https://github.com/firesim/firesim
    cd firesim
    git checkout 1.14.2
    ./scripts/machine-launch-script.sh
    ./build-setup.sh
    source sourceme-f1-manager.sh
    firesim managerinit --platform vitis
```

• RoSE Installation Begin by cloning RoSEin the project directory:



 ```
    git clone https://github.com/CobbledSteel/RoSE
    cd RoSE
    git checkout isca-ae
```


Next, within RoSE, run the setup script to set the proper environment variables. Make sure to run this script whenever starting a new interactive shell.

```
    source rose-setup.sh
```

After this is complete, run the following script to patch FireSim and Chipyard with the modifications described in Figure~\reffig:sync_diagram, and to instantiate submodules.

```
    bash soc/setup.sh
```

After this setup is complete, run the following script to build binaries for the trail-navigation controllers evaluated in Section IV for generating RISC-V Fedora images containing the controllers and ONNX models.

 ```
    bash soc/build.sh
 ```


Next, run the following script to install dependencies and configure parameters for the RoSEdeployment scripts, using the AirSim IP address provided by the authors.


    source deploy/setup.sh [AIRSIM IP]


• Optional References
This evaluation environment provides pre-built FPGA bitstreams, pre-trained controller DNN models, and AirSim environment binaries. To build new bitstreams, run the following:

 ```
    bash soc/buildbitstreams.sh
 ```

To train new classifier DNNs using the provided datasets, run the following.

```
    bash env/dnn/train.sh
```

Finally, the steps for building custom Unreal Engine maps are out of the scope of this evaluation. However, new environments can be built using the documentation provided at (https://microsoft.github.io/AirSim/build_linux/).

## Experiment Workflow

Now that the environment has been set up and the target hardware and software have been built, one can run the experiments in this work by running the following scripts`.\footnote` 
Please reach out to authors when planning to run experiments to guarantee exclusive access to FPGA resources. All the experiments can be executed by running the following command. This will generate CSV files as well as videos recorded from the front-facing camera of the simulated UAV in `deploy/logs/`. 



 ```bash
    bash deploy/scripts/run-all.sh
 ```


To run individual experiments corresponding to the figures in this work, the following scripts are also provided (which are all included in the main script). 


• Figure 10:


 ```bash
    bash deploy/scripts/tunnel-exp.sh
 ```


• Figures 15, 16:


 ```bash
    bash deploy/scripts/rose-perf-sync-only.sh
 ```



 ```bash
    bash deploy/scripts/rose-perf-tunnel-exp.sh
 ```


• Figures 11, 14:


 ```bash
     bash deploy/scripts/rose-hw-sw-sweep.sh
 ```


• Figure 12:


 ```bash
    bash deploy/scripts/rose-velocity-sweep.sh
 ```


• Figure 13:


 ```bash
    bash deploy/scripts/rose-dynamic-exp.sh
 ```


## Figures and Evaluation
After executing the prior experiments, figures can be generated using the CSV outputs by running the following command. The figures will be available as PDF files in `deploy/figures/`.



 ```python
    python3 deploy/scripts/generate-figures.py
 ```

## Experiment Customization
• Building New FPGA Images In addition to the provided SoC configurations, users can evaluate other designs. To evaluate new designs, refer to the Chipyard documentation, as well as the example RoSÉ-annotated configs found in `soc/src/main/scala/RoSEConfigs.scala`.

• Designing AirSim Environments If users install Unreal Engine as well as AirSim, it is possible to create new `maps/environments` for robot agents to interact with. By default, one can modify the `Blocks environment` provided by AirSim. Additional assets and maps can be designed by users, or obtained from the Unreal Marketplace.

• Changing Simulation Parameters RoSEprovides flags that can be used to select different simulation parameters. To view available parameters for deploying simulations, refer to `deploy/hephaestus/runner.py`. Example configurations include changing simulation granularity, or deploying a car vs a drone simulation.

Additionally, new controller ONNX models can be trained using the provided dataset and evaluated using the provided ```drone\_test``` executable. 

