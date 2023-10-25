#!/bin/bash
# set -x

# This is an example of the sort of thing you might want to do in an init script.
# Note that this script will be run exactly once on your image in qemu. 

# Note: you will see a bunch of fedora boot messages and possibly even a login
# prompt while building as this script runs. Don't worry about the login promt,
# your script is running in the background.

# In this case, we will use fedora's package manager to install something (the
# full-featured 'time' command to replace the shell builtin). We also use pip
# to install a python package used by one of the benchmarks. You can also
# download stuff, compile things that don't support cross-compilation, and/or
# configure your system in this script.

# Note that we call poweroff at the end. This is recomended because this script
# will be run automatically during the build process. If you leave it off, the
# build script will wait for you to interact with the booted image and shut
# down before it continues (which might be useful when debugging a workload).
apt update -y
apt install -y python3-pip build-essential 
pip3 install -U rosdep rosinstall_generator vcstool catkin_pkg catkin_tools
rosdep init
rosdep update
mkdir ~/ros_catkin_ws
cd ~/ros_catkin_ws
rosinstall_generator ros_comm --rosdistro noetic --deps --tar --exclude roslisp genlisp > noetic-comm.rosinstall
mkdir ./src
vcs import --input noetic-comm.rosinstall ./src
# Remove genlisp from Python lists
sed -i "/'genlisp',/d" src/ros_comm/roswtf/test/test_roswtf_command_line_offline.py
sed -i "/'genlisp',/d" src/ros_comm/roswtf/test/check_roswtf_command_line_online.py

# Modify .travis-raw.sh
sed -i 's/:genlisp//g' src/geneus/.travis-raw.sh

# Modify dependencies.dot
sed -i 's/| genlisp//g' src/catkin/doc/dependencies.dot

# Remove genlisp entries from .rosinstall files
sed -i '/genlisp.git/,+1d' src/catkin/test/checks/test-nocatkin.rosinstall
sed -i '/genlisp.git/,+1d' src/catkin/test/network_tests/test.rosinstall

# Modify CMakeLists.txt
sed -i 's/genlisp //g' src/message_generation/CMakeLists.txt

# Modify package.xml
sed -i '/<run_depend>genlisp<\/run_depend>/d' src/message_generation/package.xml

# Remove roslisp references
sed -i '/add dependency on roslisp/d' src/ros_comm/ros_comm/CHANGELOG.rst
sed -i '/<run_depend>roslisp<\/run_depend>/d' src/ros_comm/ros_comm/package.xml

# Install ros dependencies
rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y --skip-keys='python3-catkin-pkg-modules python3-rosdep-modules'

# Build ROS1
catkin build

poweroff
