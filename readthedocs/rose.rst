RoSÉ Tutorial
=============

This artifact appendix describes how to use RoSÉ to run end-to-end
robotics simulations, and how to reproduce experimental results.

The instructions assume that a user already has robotic applications and
hardware configurations developed, and provides reference examples used
in the evaluation of this work. While RoSÉ can be used to develop new
hardware and software, instructions to do so are outside of the scope of
the artifact evaluation.

Tutorial Meta-Information Checklist
-----------------------------------

-  Runtime environment: Ubuntu 18.04.6 LTS, Vitis v2021.1
-  Hardware (FireSim Simulation): Intel Xeon Gold 6242, Xilinx U250
-  Hardware (AirSim Simulation): AWS EC2 Instance (g4dn.2xlarge), Intel
   Xeon Platinum 8259CL, Tesla T4.
-  How much disk space is required?: 200GB
-  Experiments: AirSim/FireSim end-to-end full stack simulations of a
   UAV using RoSE, running DNN-based controllers. Experiments evaluate
   both UAV and simulator performance.
-  Program: Chisel (RTL), C++ (FireSim bridge drivers, robotic control
   software), Python (Synchronizer and scheduler.)
-  Quantitative Metrics: DNN Latency, mission time, average flight
   velocity, accelerator activity factor.
-  Qualitative Metrics: Flight trajectories, flight recordings.
-  Output: CSV logs from the synchronizer, tracking UAV dynamics,
   sensing requests, and control targets.
-  How much time is needed to prepare the workflow?: 4 hours (scripted
   installation).
-  How much time is needed to complete experiments?: 48 hours (scripted
   run, scripted result parsing)
-  Publicly available: Yes.
-  Code licenses: Several, see download.
-  Contact for Artifact Evaluator: Contact SLICE support
   (slice-support@eecs.berkeley.edu) if you need help setting up AWS
   instances.

Description
-----------

(1) How to access:

• The artifact consists of the core RoSÉ repos- itory, modifications to
Firesim, Chipyard, and ONNX Runtime:

-  RoSÉ Core: Deployment, synchronization, and evaluation software, as
   well as hardware configurations, and patches to FireSim and Chipyard.
   (https://github.com/CobbledSteel/RoSE)
-  FireSim: Top-level FPGA-Accelerated RTL Simulation Environment
   (https://github.com/firesim/firesim)
-  Chipyard: RISC-V SoC generation environment
   (https://github.com/ucb-bar/chipyard)
-  RISCV ONNX Runtime: Software for executing HW-accelerated DNN models,
   modified for use in UAV control
   (https://github.com/ucb-bar/onnxruntime-riscv/tree/onnx-rose).

Additionally, this evaluation builds upon the following infrastructures.
For the purpose of the evaluation, binaries for simulators built from
Unreal Engine and AirSim are provided.

-  Unreal Engine: 3D Environment development, simulation, and rendering
   platform (https://www.unrealengine.com/en-US/ue-on-github)
-  AirSim: UAV simulation plugin for Unreal Engine
   (https://github.com/microsoft/AirSim)

(2) Dependencies - Hardware

To run a full simulation with RoSÉ, access to a GPU and FPGA is
required, although these can be hosted on separate computers. For this
artifact evaluation, instructions for running simulations on a
locally-provisioned FPGA are provided. However, RoSÉ can also be used
using AWS EC2 FPGA instances (e.g. f1.2xlarge). In this artifact we
provide build scripts for generating bitstreams for locally-provisioned
FPGAs.

Additionally, GPU access is needed in order to run robotics environment
simulations with rendering. For this evaluation, AirSim binaries
packaged using Unreal Engine are provided.

To use RoSÉ in the cloud, ne AWS EC2 c5.4xlarge instance (also referred
to as “manager” instance), and one f1.2xlarge instance is required. The
latter will be launched automatically by FireSim’s manager.

(3) Dependencies - Software

Use ssh or mosh on your local machine to remote access evaluation
instances. All other requirements are automatically installed by scripts
in the following sections.

Installation
------------

To begin installation, clone the repository:

::

       git clone https://github.com/ucb-bar/RoSE.git
       cd RoSE
       git checkout isca-ae

FireSim Installation
~~~~~~~~~~~~~~~~~~~~

Begin by installing FireSim by running the following commands within the
RoSÉ repository.

::

       git submodule update --init ./soc/sim/firesim
       cd ./soc/sim/firesim
       ./scripts/machine-launch-script.sh
       ./build-setup.sh
       source sourceme-f1-manager.sh
       firesim managerinit --platform vitis

RoSÉ Installation
~~~~~~~~~~~~~~~~~

Begin by cloning RoSÉ in the project directory:

Next, within RoSÉ, run the setup script to set the proper environment
variables. Make sure to run this script whenever starting a new
interactive shell.

::

       source rose-setup.sh

After this is complete, run the following script to patch FireSim and
Chipyard to support RoSÉ, and to instantiate submodules.

::

       bash soc/setup.sh

After this setup is complete, run the following script to build binaries
for the trail-navigation controllers evaluated in Section IV for
generating RISC-V Fedora images containing the controllers and ONNX
models.

::

      bash soc/build.sh

Next, run the following script to install dependencies and configure
parameters for the RoSÉ deployment scripts, using the IP address of the
GPU system that will be used to run the provided AirSim binaries.

::

   source deploy/setup.sh [AIRSIM IP]

Bitstream Generation
~~~~~~~~~~~~~~~~~~~~

To build bitstreams for Rocket+Gemmini and BOOM+Gemmini configurations,
run the following.

::

      bash soc/buildbitstreams.sh

DNN Training
~~~~~~~~~~~~

This artifact provides pre-trained models for evaluation. To train new
classifier DNNs using the provided datasets, run the following,
selecting between the given ResNet configurations. Each training run
will output an ONNX model named ``trail_dnn_resnet[xy].onnx``.

::

       bash env/train/train_resnet.py (6|11|14|18|34|50)

Finally, the steps for building custom Unreal Engine maps are out of the
scope of this evaluation. However, new environments can be built using
the documentation provided at
(https://microsoft.github.io/AirSim/build_linux/).

Experiment Workflow
-------------------

Now that the environment has been set up and the target hardware and
software have been built, one can run the experiments in this work by
launching an AirSim simulation and running the following scripts. All
the experiments can be executed by running ``run-all.sh``. This will
generate CSV files as well as videos recorded from the front-facing
camera of the simulated UAV in ``deploy/hephaestus/logs/``.

.. code:: bash

      bash deploy/scripts/run-all.sh

To run individual experiments corresponding to the figures in this work,
the following scripts are also provided (which are all included in the
main script).

• Figure 10:

.. code:: bash

      bash deploy/scripts/tunnel-exp.sh

• Figures 15, 16:

.. code:: bash

      bash deploy/scripts/rose-perf-sync-only.sh

.. code:: bash

      bash deploy/scripts/rose-perf-tunnel-exp.sh

• Figures 11, 14:

.. code:: bash

       bash deploy/scripts/rose-hw-sw-sweep.sh

• Figure 12:

.. code:: bash

      bash deploy/scripts/rose-velocity-sweep.sh

• Figure 13:

.. code:: bash

      bash deploy/scripts/rose-dynamic-exp.sh

Figures and Evaluation
----------------------

After executing the prior experiments, figures can be generated using
the CSV outputs by running the following command. The figures will be
available as in ``deploy/figures/``.

.. code:: python

      python3 deploy/scripts/generate-figures.py

Experiment Customization
------------------------

• Building New FPGA Images In addition to the provided SoC
configurations, users can evaluate other designs. To evaluate new
designs, refer to the Chipyard documentation, as well as the example
RoSÉ-annotated configs found in
``soc/src/main/scala/RoSEConfigs.scala``.

• Designing AirSim Environments If users install Unreal Engine as well
as AirSim, it is possible to create new ``maps/environments`` for robot
agents to interact with. By default, one can modify the
``Blocks environment`` provided by AirSim. Additional assets and maps
can be designed by users, or obtained from the Unreal Marketplace.

• Changing Simulation Parameters RoSEprovides flags that can be used to
select different simulation parameters. To view available parameters for
deploying simulations, refer to ``deploy/hephaestus/runner.py``. Example
configurations include changing simulation granularity, or deploying a
car vs a drone simulation.

Additionally, new controller ONNX models can be trained using the
provided dataset and evaluated using the provided ``drone_test``
executable.
