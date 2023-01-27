# EducationProject for Franka panda with impedance control

This repository is forked from: https://github.com/1am5hy/EducationProject.git

## Install 
pip install -e .

## Examples
python curi_sim/test_panda.py

## Plot the contact force
1. python curi_sim/test_panda.py
2. sudo apt install ros-$ROS_DISTRO-plotjuggler
3. source ~/catkin_ws/devel/setup.bash
4. roscore
5. Open a new terminal, source ~/catkin_ws/devel/setup.bash
6. rosrun plotjuggler plotjuggler
7. select "ROS Topic Subscriber" in the left column, and then "Start"
8. select /contact_force and /object_velocity
9. Drag the data of both topic to the right display box