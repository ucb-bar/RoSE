---
layout: page
permalink: /tutorial/
title: tutorial 
description: 
nav: true
nav_order: 3
---

## RoSÉ Tutorial at MICRO 2023:
**Designing, Deploying, and Evaluating Full-Stack Robotics Systems With RoSÉ**

### Overview
We are running a half-day tutorial on RoSÉ, an open-source hardware-software co-simulation infrastructure that enables the evaluation of robotics system-on-chips (SoCs). 

Using RoSÉ, users will be able to deploy end-to-end simulations of a robotics SoC running in a simulated environment. By co-simulating a cycle-exact RTL simulation with a robotics simulator, users can evaluate the performance of their SoC, capturing performance metrics of both CPU cores and hardware accelerators, and gain system level insights about architectural choices. Users will also be able to evaluate how their SoC can perform in differing maps, sensor configurations, and tasks by using OpenAI Gym (with examples using AirSim and MuJoCo) to construct a robot's simulated environment. Finally, users will be able to explore different software choices for their robotics application, including both DNN-based and classical robotics workloads. 

By completing this tutorial, users will be able to use RoSÉ to evaluate the end-to-end performance of their robotics SoC through customized experiments.

A tentative schedule is available below.

### Logistics and Registration

Venue: MICRO 2023 (will be held in-person)

Date: Sunday, October 29th, 2023

Time: 8 AM - 12 PM EDT

**Registration**
- Please register to attend [MICRO](https://microarch.org/micro56/index.php).
- Additionally, please fill out [this Google form](https://forms.gle/5bqrZmRYvmHZC3RZ9) if you would like to attend our tutorial.

### Schedule 

<table>
  <thead>
    <tr>
      <th>Time</th>
      <th>Activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>8:00-8:30</td>
      <td>Introduction</td>
    </tr>
    <tr>
      <td>8:30-8:45</td>
      <td>RoSE Infrastructure</td>
    </tr>
    <tr>
      <td>8:45-9:00</td>
      <td>Coffee Break</td>
    </tr>
    <tr>
      <td>9:00-9:50</td>
      <td>Interactive Activity 1: Building a Complete RoSE Application</td>
    </tr>
    <tr>
      <td>9:50-10:00</td>
      <td>Coffee Break</td>
    </tr>
    <tr>
      <td>10:00-10:20</td>
      <td>Sensor Modeling in RoSE with CoSMo</td>
    </tr>
    <tr>
      <td>10:20-11:00</td>
      <td>Interactive Activity 2: Advanced SW Flow with RoSE</td>
    </tr>
    <tr>
      <td>11:00-11:15</td>
      <td>Coffee Break</td>
    </tr>
    <tr>
      <td>11:15-12:00</td>
      <td>Demos, Conclusion, and Q&A</td>
    </tr>
  </tbody>
</table>


# RoSÉ Tutorial Instructions: MICRO 2023

## Prerequsites
Welcome to the RoSÉ tutorial! The interactive contents of the tutorial are provided in two formats:

### 1) AWS instance (Recommended): To access the AWS instance
 * Navigate to the URL Provided at the tutorial
 * Put down your initials in column B to claim an instance 
 * Follow the link provided. (NOTE: If prompted with a certificate/security warning, accept your browser's conditions to proceed)
 * Login with the following details
     * Username: ubuntu
     * Password: rose2023
 * Click on the RoSE icon in the sidebar to open the RoSE terminal session you will be using for the rest of the tutorial
 * Note: To paste copied text into the session, click the clipboard icon in the top left and paste it into the text box. This copies your clipboard into the AWS clipboard

### 2) VirtualBox: To follow the instructions offline/after the tutorial, we provide flash drives with VirtualBox images
 * Download and install VirtualBox 6.1 or newer
 * Open VirtualBox and select File > Import Appliance
 * Select `RoSE\_MICRO\_Tutorial.ova` and follow the instructions to import the image. Ensure you have sufficient disk space (64GB+) for storing the extracted contents. 
 * Note: The image has been set to use 8GB of RAM to run most of the exercises. If you wish to run the RISC-V Linux image at the end of the tutorial, ensure to allocate at least 20GB
 * Launch the VirtualBox image
 * Login with the following details
     * Username: rose
     * Password: rose2023
 * Click on the RoSE icon in the sidebar to open the RoSE terminal session you will be using for the rest of the tutorial

We hope you enjoy the tutorial! We encourage you to ask any questions!

Throughout the tutorial we have provided convenient commands for executing the experiments via `rose [cmd]`. Please refer to the notes for more detailed commands, locations of relevant files, and discussion.

## Interactive Section 1: End-to-end RoSE flow with a CartPole/InvertedPendulum Robot

### 1) Run a RoSE simulation
 * Ensure you have a RoSE terminal open by clicking on the RoSE icon in the sidebar.
     * Note: Alternatively this can be achieved by navigating to the RoSE root (`~/RoSE/Desktop/`) and running `source rose-setup.sh`
 * In your RoSE terminal, run `rose run`
     * You should see an image open depicting the InvertedPendulum, and a plot of the observation/action measurements. This simulation will end if the pendulum loses balance (moves past 0.2 radians) or the simulation time reaches 0.5s. 
     * After the simulation is complete, a VLC window will open with a recording of the trajectory, followed by a plot.
     * Note: If using AWS, this can take slightly longer for the first execution 
     * Note: To run the simulation for scripting, use `python3 ./deploy/hephaestus/rose.py` 
     * Note: To directly view the logs, run `rose view logs` or navigate to `./deploy/hephaestus/logs/`. 

### 2) Introduce sensor latency to the scenario
 * In your RoSE terminal, run `rose edit gym_conf`. This will open a text editor window.
 * Modify the field `latency` (line 5) to be `0.05`. This introduces a delay of 50ms to the observation data.
 * Save the file, and then run `rose run` to start a new simulation
 * Note: Observe the arrival time of the first action. Does this match the latency? If there are differences, what might contribute? 
 * Note: How does the action throughput compare to the previous experiment? What might this imply about how the SoC processes sensor data
 * Note: Many real applications have varying latency and bandwidths for different sensor modalities (e.g. fast inertial data, but slow GPS data). To model such scenarios, consider defining multiple packets that access a subset of the observation space.

### 3) (Optional) View and modify SoC software.
 * Navigate to `./soc/sw/baremetal/`. This is a build directory for baremetal code examples to run on the SoC
 * To view the current code, open `inverted-pendulum-pid.c`. Note that the `read_pid_obs` function is blocking; after making an observation request, it continues to poll until a packet is returned to the SoC
 * To view a non-blocking example, open `inverted-pendulum-pid-nonblock.c`
 * To rebuild the binaries and update them within RoSE, run `make` and `make install`
 * Return to the RoSE root directory (`~/Desktop/RoSE`)
 * To modify which software is being executed by RoSE, run `rose edit firesim_conf`
     * This file configures the FireSim simulation used by RoSE. To modify tha software being executed on the SoC, navigate to to `workload.workload_name` (line 71)
     * To run code defined in `[program].c`, set this to be `[program].json` (This file is automatically generated when you run `make_install` earlier). As an optional exercise, set this to `inverted-pendulum-pid-nonblock.json` to see how the action throughput is affected by using nonblocking sensor IO. 
 * If you made any modifications, such as building new SW or changing the SW configuration, run `rose build`. This updates the RTL simulation to use the new software.
     * Note: For FireSim users, this is equivalent to `firesim infrasetup`  
 * To view console output from your simulation, after running `rose run`, execute `screen -r fsim0` in a new terminal window to view the live UART output from the SoC containing `stdio`. 
### 4) Modify the SoC's hardware configurations 
 * Run `rose gym_conf` and reset the packet latency to 0.
 * Modify the clock frequency
     * Run `rose edit sim_conf`
     * Set `firesim_freq` to 200_000, setting the SoC's clock frequency to 200KHz
     * Save the file
     * Run `rose run` to start a new simulation
         * Note: Does the simulation run for the full duration? 
     * (Optional) Set the SoC's clock frequency to 400MHz and start a new simulation. What behavior do you observe in the robot?
 * (Optional) Modify the SoC architecture
     * Currently the system uses an in-order Rocket Core. Suppose you want to improve the performance of the system by switching to a more powerful CPU.
     * Run `rose edit firesim_conf` and navigate to `target_config.default_hw_config` (Line 49). Currently, this points to a Rocket configuration; to switch to a superscalar BOOM core, comment on this line and uncomment the line containing BOOM below.
     * To use this configuration in your RoSE simulation, save the file and run `rose build`
     * Note: This workload is heavily bottlenecked by IO. Is switching to a more powerful core expected to improve performance? Do simulation results match your expectations?
     * Note: (FireSim users) To view or add new HW configs, edit `./soc/sim/config/config_hwdb_local.yaml`. 
     * Note: (FireSim users) Scala files for RoSE-enabled configurations are located in `./soc/src/main/scala/RoSEConfigs.scala` for configs that include the RoSE TileLink interface, and `./soc/src/main/scala/RoSEFireSimConfigs.scala` for configs that enable the RoSE bridges.

## Interactive Section 2: Full-stack SW flows 

The previous section describes the basics needed to run an end-to-end simulation with RoSE. However, most interesting workloads involve a more substantial SW/HW stack. This section defines how to build and configure ROS1, a widely-used framework for robotics applications. Next, this section describes the software flow for DNN-based robotics applications, including the trail navigation experiments in the RoSE paper. 

### 1) Launching RISC-V Ubuntu 20.04 with ROS1 Noetic for your SoC 
* Note: If using VirtualBox, ensure you have at least 20 GB of memory allocated for this step or the QEMU session cannot start. 
* Ensure you have a RoSE terminal open by clicking on the RoSE icon in the sidebar.
* Run `rose vm ros` 
 * This launches a RISC-V QEMU session allowing interactive access to the Linux image that is deployed to the SoC.
 * Note: This is equivalent to the following FireMarshal command: `marshal launch ./soc/sw/rose-images/ros-ubuntu.json`
* After the image finishes booting, you will see a login prompt. Enter the username `root` and the password `firesim`. 
  * Note: Ubuntu might print diagnostic information on top of the login prompt. However, you can still log in as usual. 
* (Optional) run `uname -a` to view system information
* Navigate to the ROS1 workspace: `cd ./ros_catkin_ws/`. This contains the ROS1 build and installation directory
* To set the ROS environment variables, run `source ./devel/setup.bash`
* To start the ROS server, run `roscore` to validate that it's functional. Terminate with `^C`
* (Optional) Build a new ROS package from the source:
  * TODO hector_mapping
* Exit the QEMU session by running `poweroff`
* (Optional) For more information on how to build workloads, please refer to the FireMarshal documentation: firemarshal.readthedocs.io/en/latest/

### 2) (Optional) Modify the build flow for the ROS1 image
 * Run `ls ./soc/sw/rose-images/ros-ubuntu/overlay/root/`. This contains any files that will be copied into the root directory when building your Linux image
 * Open `./soc/sw/rose-imges/ros-ubuntu/guest-init.sh`. This contains scripts that are executed when building the Linux image within the emulated RISC-V session.
     * Note: Since ROS1 is typically natively installed on a system with access to the system's package manager, we use a native build flow. All commands needed to build the core ROS system are in this file. 
     * Note: We follow ROS' source-build flow; for more detailed information, refer to the documentation: wiki.ros.org/noetic/Installation/Source. ROS1's lisp integration is not well supported in RISC-V, so we patch this out. If you need to use lisp, please reach out to the tutorial organizers for the bootstrapping process. 
 * Open `./soc/sw/rose-imges/ros-ubuntu/guest-init.sh`. This contains scripts that are executed by the host building the images. Since we use a native build flow, this is unused. However, if using cross-compilation, relevant build scripts should be placed here.
 * To rebuild a new image, run `marshal build ./soc/sw/rose-images/ros-ubuntu.json`. Remember to run `rose build` afterward to update the newly updated image within RoSE. 

### 3) Visualize trail-navigation DNNs. 
 * To hardware-accelerate DNN-based flows, we use the Gemmini systolic-array accelerator. We deploy DNN applications using the ONNX Runtime environment. 
 * Run `rose view dnn 14` to launch a visualization of the ResNet14 model used in the trail navigation experiments in the RoSE paper. 
     * (Optional) View other architectures, such as ResNet6 or ResNet34
 * (Optional) Open `./env/train/train_resnet.py` and go to line 248 to view how to export to a gemmini-compatible ONNX model with PyTorch

### 4) Evaluate the performance and accuracy of trail-navigation DNNs
 * Evaluate ResNet14 running on a CPU using the spike RISC-V ISA simulator by: `rose dnn cpu 14`. View the cycle count estimate and the classification results.
 * Evaluate ResNet14 using spike's Gemmini model: `rose dnn gemmini 14`. How does the performance compare to the CPU implementation?
 * Run ResNet6 on Gemmini with `rose dnn gemmini 6`, and compare the performance and the quality of the prediction. How about ResNet34?

