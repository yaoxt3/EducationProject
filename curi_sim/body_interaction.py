#!/usr/bin/env python3
"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.005" />
    <worldbody>
        <body name="robot" pos="0.1 2.5 0.15">
            <joint axis="1 0 0" damping="0.1" name="slide0" pos="0 0 0" type="slide"/>
            <joint axis="0 1 0" damping="0.1" name="slide1" pos="0 0 0" type="slide"/>
            <joint axis="0 0 1" damping="1" name="slide2" pos="0 0 0" type="slide"/>
            <geom mass="10.0" pos="0 0 0" rgba="1 0 0 1" size="0.3 0.2 0.15" type="box"/>
			<camera euler="0 0 0" fovy="40" name="rgb" pos="0 0 2.5"></camera>
        </body>
        
        <body name="object" pos="0.1 1 0.2" quat="1 0 0 0" >
            <inertial pos="0 0 0" mass=".155" diaginertia="0.0003686049833333333 0.001255996129167 0.001460916979167" />
            <joint axis="1 0 0" name="cylinder:slidex" type="slide"/>
            <joint axis="0 1 0" name="cylinder:slidey" type="slide"/>
            <geom name="object" type="box" size="0.3 0.2 0.15" group="1" rgba=".9 0 0 1" />
            <site name="obj_front" pos="-0.15074 0 0" quat="1 0 0 0"          type="box" size="1e-5 0.04 0.02" group="2" rgba="1 1 0 1" />
            <site name="obj_right" pos="0 -0.07449 0" quat="0.707 0 0 0.707"  type="box" size="1e-5 0.10 0.02" group="2" rgba="1 1 0 1" />
            <site name="obj_left"  pos="0 +0.07449 0" quat="0.707 0 0 -0.707" type="box" size="1e-5 0.10 0.02" group="2" rgba="1 1 0 1" />
            <site name="obj_rear"  pos="+0.15074 0 0" quat="0 0 0 1"          type="box" size="1e-5 0.04 0.02" group="2" rgba="1 1 0 1" />
            <!-- Point contact sites -->
            <!-- <site name="obj_front" pos="-0.15075 0 0" quat="1 0 0 0"          type="sphere" size="1e-3" group="2" rgba="1 1 0 1" /> -->
            <!-- <site name="obj_right" pos="0 -0.0745 0"  quat="0.707 0 0 0.707"  type="sphere" size="1e-3" group="2" rgba="1 1 0 1" /> -->
            <!-- <site name="obj_left"  pos="0 +0.0745 0"  quat="0.707 0 0 -0.707" type="sphere" size="1e-3" group="2" rgba="1 1 0 1" /> -->
            <!-- <site name="obj_rear"  pos="+0.15075 0 0" quat="0 0 0 1"          type="sphere" size="1e-3" group="2" rgba="1 1 0 1" /> -->
        </body>
    
        
        
        <body name="box" pos="0.1 -2 0.15">
            <joint axis="1 0 0" damping="0.1" name="slide3" pos="0 0 0" type="slide"/>
            <joint axis="0 1 0" damping="0.1" name="slide4" pos="0 0 0" type="slide"/>
            <joint axis="0 0 1" damping="1" name="slide5" pos="0 0 0" type="slide"/>
            <geom mass="10.0" pos="0 0 0" rgba="1 0 0 1" size="0.3 0.2 0.15" type="box"/>
        </body>
        
        <body name="floor" pos="0 0 0.025">
            <geom condim="3" size="5.0 10.0 0.02" rgba="0 1 0 1" type="box"/>
        </body>
    </worldbody>
    <actuator>
        <motor gear="2000.0" joint="slide0"/>
        <motor gear="2000.0" joint="slide1"/>
        <motor gear="2000.0" joint="slide3"/>
        <motor gear="2000.0" joint="slide4"/>
    </actuator>
</mujoco>
"""

# <body name="slider" pos="0.1 1 0.2">
#             <geom mass="1" size="0.15 0.15 0.15" type="box"/>
#             <joint axis="1 0 0" name="cylinder:slidex" type="slide"/>
#             <joint axis="0 1 0" name="cylinder:slidey" type="slide"/>
#         </body>

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
phase1 = 0
phase2 = 0
while True:
    # sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
    # sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
    # sim.data.ctrl[2] = math.cos(t / 10.) * 0.01
    # sim.data.ctrl[3] = math.sin(t / 10.) * 0.01
    if phase1 < 100:
        sim.data.ctrl[1] = -0.1
        sim.data.ctrl[3] = -0.1
        phase1 += 1
    # else:
    #     sim.data.ctrl[1] = 0
    elif phase2 < 100:
        sim.data.ctrl[1] = 0.1
        sim.data.ctrl[3] = 0.1
        phase2 += 1
    else:
        phase1 = 0
        phase2 = 0
    t += 1
    sim.step()
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break
