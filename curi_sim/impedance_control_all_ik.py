import math
import robots
import environments
from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco_py as mjp
import numpy as np
from time import sleep
from controller import Dynamics, JointPositionImpedanceGenerator
import rospy
from std_msgs.msg import String, Float32
import threading
import time
from utils.viewer_utils import render_frame
from utils.tf import quatdiff_in_euler

# ----------- Inverse Kinematics -----------
from dm_control import mujoco
from dm_control.mujoco.testing import assets
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import inverse_kinematics as ik
mjlib = mjbindings.mjlib

_ARM_XML = assets.get_contents('/home/yxt/Research/code/EducationProject/curi_sim/description/Franka/arm_for_IK.xml')
_SITE_NAME = 'ee_joint'
_JOINTS = ['panda0_joint1', 'panda0_joint2', 'panda0_joint3', 'panda0_joint4', 'panda0_joint5', 'panda0_joint6', 'panda0_joint7']
_TOL = 1.2e-8
_MAX_STEPS = 150
_MAX_RESETS = 10
# -----------------------------------------

# joint limitation
joint_max = [2.8973, 1.7628, 2.8973, -0.4, 2.8973, 2.1127, 2.8973]
joint_min = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -1.6573, -2.8973]
    
# set starting joint position
def ik_set_joint_pos(physics, joint_pos):
    physics.named.data.qpos[_JOINTS] = joint_pos


def inverse_kinematics(target_pos, target_quat, current_pos):
    physics = mujoco.Physics.from_xml_string(_ARM_XML)
    count = 0
    physics2 = physics.copy(share_model=True)
    ik_set_joint_pos(physics2, current_pos)
    step = 0
    while True:
        result = ik.qpos_from_site_pose(
            physics=physics2,
            site_name=_SITE_NAME,
            target_pos=target_pos,
            target_quat=target_quat,
            joint_names=_JOINTS,
            tol=_TOL,
            max_steps=_MAX_STEPS,
            inplace=False,
        )
        print(f"step:{step}, result:{result}")
        if result.success:
            break
        elif count < _MAX_RESETS:
            ik_set_joint_pos(physics2, current_pos)
            count += 1
        else:
            raise RuntimeError(
                'Failed to find a solution within %i attempts.' % _MAX_RESETS)
        step += 1
              
    return result.qpos


# inverse kinematics thread
def inverse_kinematics_thread():
    global ik_physics2, ik_target_ee_pos, ik_target_ee_quat, ik_current_jpos, ik_result, ik_solver
    
    while True:
        if ik_solver:
            count = 0
            ik_physics2.named.data.qpos[_JOINTS] = ik_current_jpos
            while True:
                result = ik.qpos_from_site_pose(
                    physics=ik_physics2,
                    site_name=_SITE_NAME,
                    target_pos=ik_target_ee_pos,
                    target_quat=ik_target_ee_quat,
                    joint_names=_JOINTS,
                    tol=_TOL,
                    max_steps=_MAX_STEPS,
                    inplace=False,
                )
                if result.success:
                    break
                elif count < _MAX_RESETS:
                    ik_physics2.named.data.qpos[_JOINTS] = ik_current_jpos
                    count += 1
                else:
                    raise RuntimeError(
                        'Failed to find a solution within %i attempts.' % _MAX_RESETS)
            ik_result = result.qpos.copy()
            
        time.sleep(0.01)


# --------- Modify as required ------------ #
# Task-space controller parameters
# stiffness gains
P_pos = 1800.
P_ori = 600.
# damping gains
D_pos = 2.*np.sqrt(P_pos)
D_ori = 2.
# ---------------------------------------- #

def compute_ts_force(curr_pos, curr_ori, goal_pos, goal_ori, curr_vel, curr_omg, goal_vel):
    delta_pos = (goal_pos - curr_pos).reshape([3, 1])
    delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3, 1])
    F = np.vstack([P_pos*(delta_pos), P_ori*(delta_ori)]) - \
        np.vstack([D_pos*(curr_vel).reshape([3, 1]),
                   D_ori*(curr_omg).reshape([3, 1])]) 

    error = np.linalg.norm(delta_pos) + np.linalg.norm(delta_ori) 

    return F, error


def impedance_control_integration(ctrl_rate):
    global env, target_pos, target_vel, original_ori, joint_controller, robot, run_controller
    global ik_target_ee_pos, ik_target_ee_quat, ik_current_jpos, ik_result, ik_solver
    count = 0
    threshold = 0.0000005

    target_pos = curr_ee.copy()
    target_vel = curr_vel_ee.copy()
    step = 1
    switch_controller = 1

    kp = 1000
    kd = 4.1
    target_joint = []
    target_ee_ori = []
    while run_controller:
        error = 100.
        force_norm = 0
        total_force = 0
        while error > threshold:
            now_c = time.time()
            curr_pos, curr_ori = env.get_ee_pose()
            curr_vel, curr_omg = env.get_ee_velocity() # get linear velocity and angular velocity
            ee_vel = np.sqrt(np.sum(np.square(curr_vel)))
            pub_robot_vel.publish(ee_vel)

            target_vel[1] = y_target_vel
            delta_vel = y_target_vel * env.sim.model.opt.timestep * step
            if switch_controller == 1:
                target_pos[1] = curr_ee[1] + delta_vel

            joint_pos = env.get_site_pos("ee_joint")
            object_pos = env.get_site_pos("obj_contact")
            dis = np.sqrt(np.sum(np.square(joint_pos[1] - object_pos[1])))
            pub_d.publish(dis)

            vel = env.joint_velocities()[7]
            pub_vel.publish(vel)
            
            # updata ik solver parameters
            ik_target_ee_pos = curr_pos.copy()
            ik_current_jpos = env.joint_position()[:7]

            for i in range(env.sim.data.ncon):
                # Note that the contact array has more than `ncon` entries,
                # so be careful to only read the valid entries.
                contact = env.sim.data.contact[i]
                geom2_body = env.sim.model.geom_bodyid[env.sim.data.contact[i].geom2]
                # Use internal functions to read out mj_contactFor
                c_array = np.zeros(6, dtype=np.float64)
                mjp.functions.mj_contactForce(env.sim.model, env.sim.data, i, c_array)
                contact1 = env.sim.model.geom_id2name(contact.geom1)
                contact2 = env.sim.model.geom_id2name(contact.geom2)
                if contact1 == "link7_col" or contact2 == "link7_col":
                    force_norm = np.sqrt(np.sum(np.square(c_array[0:3])))
                else:
                    force_norm = 0
                 
            pub_f.publish(force_norm)
            total_force += force_norm
            
            while force_norm > 0:
                if count == 0:
                    target_joint = env.joint_position()[:7]
                    delta_target = 0.38 - curr_ee[1]
                    K_max = force_norm / delta_target
                    print("delta_target", delta_target)
                    print("K_max", K_max)
                    print("curr_ee", curr_ee)
                    switch_controller = 2
                count += 1
                break
            
            k_tmp = 200 + step**2
            # if object's velocity < 0.02, stop the robot
            while vel < 0.01 and switch_controller == 2:
                target_joint = ik_result.copy()
                switch_controller = 3
                print("object stop")
                break

            if count > 0: # controller 1, in-contact phase
                # end effector's pose adjustment
                ik_solver = True # start ik solver thread
                if ik_result is not None:
                    adjust_joint_pos_control = ik_result.copy()
                else:
                    adjust_joint_pos_control = env.joint_position()[:7]
                position_error_control1 = adjust_joint_pos_control - env.joint_position()[:7]
                
                # impedance control based on joint position
                position_error = (target_joint - env.joint_position()[:7]) * 0.02
                vel_pos_error = -env.joint_velocities()[:7]
                desired_torque_control1 = (np.multiply(np.array(position_error_control1), np.array(200)) + np.multiply(vel_pos_error, kd))
                desired_torque = (np.multiply(np.array(position_error), np.array(k_tmp)) + np.multiply(vel_pos_error, kd)) + desired_torque_control1
            
            else: # controller 2, pre-move phase
                F, error = compute_ts_force(curr_pos, curr_ori, target_pos, original_ori, curr_vel, curr_omg, target_vel)
                desired_torque = np.dot(env.get_ee_jacobian("ee_joint").T, F).flatten().tolist()
                _, target_ee_ori = env.get_ee_pose() # record the ee position and pose at the moment of contact
                ik_target_ee_quat = target_ee_ori.copy()
                ik_target_ee_pos = curr_pos.copy()
                ik_current_jpos = env.joint_position()[:7]
                

            # Return desired torques plus gravity compensations
            MassMatrix = joint_controller.dynamics.MassMatrix()
            desired_torque = MassMatrix.dot(desired_torque) + joint_controller.dynamics.gravityforces(robot.joint_pos)
            robot.set_joint_torque(desired_torque)

            env.sim.step()
            
            # adjust end effector pose via inverse kinematics
            if count > 0:
                diff_ori = quatdiff_in_euler(curr_ori, target_ee_ori)
                ee_curr_pos, ee_curr_ori = env.get_ee_pose()
                if (np.absolute(diff_ori) > 0.2).any():
                    print(f"diff ori:{diff_ori}")
                    adjust_joint_pos = inverse_kinematics(ee_curr_pos, target_ee_ori, env.joint_position()[:7])
                    env.set_initial_pos(adjust_joint_pos)

            elapsed_c = time.time() - now_c
            sleep_time_c = (1. / ctrl_rate) - elapsed_c
            if sleep_time_c > 0.0:
                print(f"sleep_time_c:{sleep_time_c}")
                time.sleep(sleep_time_c)
            # time.sleep(0.001)
            step += 1


def go_to_initial_pos():
    global env, robot, joint_controller, su_flag, update_ee
    
    while update_ee:
        jtor, su_flag = joint_controller.impedance_controller_joint()
        robot.set_joint_torque(jtor)
        env.sim.step()
        sleep(0.002)


if __name__ == "__main__":
    model = load_model_from_path("./description/Franka/dynamic_franka_without_gripper_with_object.xml")
    sim = MjSim(model=model)
    env = environments.Base_env(sim)
    robot = robots.Franka_robot(sim)
    joint_controller = JointPositionImpedanceGenerator(1000, 4.1, robot)
    target1 = np.array([0.0, 0.9, 0.0, -1.7, -1.62, 1.6, 0.0])  # target 1
    target2 = np.array([0.0, 0.9, -0.4, -1.7, -1.62, 1.95, 0.3])  # target 2
    joint_controller.set_target(target2)
    initial_pos = np.array([0.0, 0.85, -0.2, -1.7, -1.62, 1.6, 1.0])
    env.set_initial_pos(initial_pos)
        
    # inverse kinematics solver
    ik_target_ee_pos = None
    ik_target_ee_quat = None
    ik_current_jpos = None
    ik_result = None
    ik_solver = False
    ik_physics = mujoco.Physics.from_xml_string(_ARM_XML)
    ik_physics2 = ik_physics.copy(share_model=True)

    rospy.init_node('talker', anonymous=True)
    pub_d = rospy.Publisher('distance', Float32, queue_size=10)  # the distance between object and robot's end effector
    pub_f = rospy.Publisher('contact_force', Float32, queue_size=10)  # the contact force
    pub_vel = rospy.Publisher('object_velocity', Float32, queue_size=10)
    pub_robot_vel = rospy.Publisher('robot_velocity', Float32, queue_size=10)
    
    update_ee = True
    su_flag = False
    update_ee_thread = threading.Thread(target=go_to_initial_pos) # go to initial position
    update_ee_thread.start()
        
    force_norm = 0
    break_flag = False
    dis = 100
    print("applied force!")
    while True:
        if su_flag:
            joint_pos = env.get_site_pos("ee_joint")
            object_pos = env.get_site_pos("obj_contact")
            dis = np.sqrt(np.sum(np.square(joint_pos - object_pos)))
            if dis > 1:
                env.sim.data.xfrc_applied[env.sim.model.body_name2id("object"), :] = np.array([0, 40, 0, 0, 0, 0])
            else:
                env.sim.data.xfrc_applied[env.sim.model.body_name2id("object"), :] = np.array([0, 0, 0, 0, 0, 0])
            
            vel = env.joint_velocities()[7] 
            pub_vel.publish(vel)
            pub_f.publish(force_norm)
            
            if dis < 0.3:
                break_flag = True
            
        if break_flag:
            break        
        env.sim.step()
        env.render()
        sleep(0.002)

    if dis <= 0.3:
        update_ee = False
        update_ee_thread.join()
        run_controller = True
    else:
        run_controller = False

    ctrl_rate = 1 / env.sim.model.opt.timestep
    render_rate = 100
    curr_ee, original_ori = env.get_ee_pose()  # end effector's pose
    curr_vel_ee, curr_omg_ee = env.get_ee_velocity()  # end effector's velocity
    target_pos = curr_ee.copy()
    target_y_vel = np.linspace(curr_vel_ee[1], curr_vel_ee[1] + 0.8, 60).tolist()
    y_target_vel = curr_vel_ee[1]
    target_vel = curr_vel_ee.copy()

    ctrl_thread = threading.Thread(target=impedance_control_integration, args=[ctrl_rate])  # multi-thread
    ctrl_thread.start()
    
    ik_thread = threading.Thread(target=inverse_kinematics_thread)
    ik_thread.start()
    
    now_r = time.time()
    i = 0

    while i < len(target_y_vel):

        y_target_vel = target_y_vel[i]
        robot_pos, robot_ori = env.get_ee_pose()
        render_frame(env.view, robot_pos, robot_ori)
        render_frame(env.view, target_pos, original_ori, alpha=0.2)
        elapsed_r = time.time() - now_r
        # if elapsed_r >= 0.0001:
        #     i += 1
        #     now_r = time.time()
        i += 1
        env.render()

    print("Done controlling. Press Ctrl+C to quit.")
    while True:
        robot_pos, robot_ori = env.get_ee_pose()
        render_frame(env.view, robot_pos, robot_ori)
        render_frame(env.view, target_pos, original_ori, alpha=0.2)
        env.render()

    run_controller = False
    ctrl_thread.join()