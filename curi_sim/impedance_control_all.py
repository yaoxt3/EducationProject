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


# --------- Modify as required ------------
# Task-space controller parameters
# stiffness gains
P_pos = 1500.
P_ori = 450.
# damping gains
D_pos = 2.*np.sqrt(P_pos)
D_ori = 2.
# -----------------------------------------

def compute_ts_force(curr_pos, curr_ori, goal_pos, goal_ori, curr_vel, curr_omg, goal_vel):
    delta_pos = (goal_pos - curr_pos).reshape([3, 1])
    delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3, 1]) # 这个是计算角度的不同，可能输入变量不是3×1的矩阵
    # delta_vel = (goal_vel - curr_vel).reshape([3, 1])
    # print
    F = np.vstack([P_pos*(delta_pos), P_ori*(delta_ori)]) - \
        np.vstack([D_pos*(curr_vel).reshape([3, 1]),
                   D_ori*(curr_omg).reshape([3, 1])]) #变成列向量堆叠在一起

    # print(f"force:{F}, shape:{np.shape(F)}")
    error = np.linalg.norm(delta_pos) + np.linalg.norm(delta_ori) #这个是求范数

    return F, error

#
def impedance_control_integration(ctrl_rate):
    global env, target_pos, target_vel, original_ori, joint_controller, robot
    count = 0
    threshold = 0.0000005

    target_pos = curr_ee.copy()
    target_vel = curr_vel_ee.copy()
    step = 1
    jpos_in_contact = []
    switch_controller = 1

    kp = 1000
    kd = 4.1
    target_joint = []
    while run_controller:
        error = 100.
        force_norm = 0
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
            print(f"object velocity:{vel}")
            pub_vel.publish(vel)

            for i in range(env.sim.data.ncon):
                # Note that the contact array has more than `ncon` entries,
                # so be careful to only read the valid entries.
                contact = env.sim.data.contact[i]
                geom2_body = env.sim.model.geom_bodyid[env.sim.data.contact[i].geom2]
                # Use internal functions to read out mj_contactFor
                c_array = np.zeros(6, dtype=np.float64)
                mjp.functions.mj_contactForce(env.sim.model, env.sim.data, i, c_array)
                if env.sim.model.geom_id2name(contact.geom1) == "link7_col" or \
                        env.sim.model.geom_id2name(contact.geom2) == "link7_col":
                    force_norm = np.sqrt(np.sum(np.square(c_array[0:3])))
                else:
                    force_norm = 0
            pub_f.publish(force_norm)

            print("Force Norm", force_norm)
            while force_norm > 0:
                if count == 0:
                    target_joint = env.joint_position()[:7]
                    switch_controller = 2
                count += 1
                break

            # if object's velocity < 0.02, stop the robot
            while vel < 0.02 and switch_controller == 2:
                target_joint = env.joint_position()[:7]
                switch_controller = 3
                break

            if count > 0: # controller 1, in-contact phase
                print(f"target_joint: {target_joint}")
                dx = (target_joint - env.joint_position()[:7]) * 0.02
                desired_qpos = env.joint_position()[:7] + dx
                position_error = desired_qpos - env.joint_position()[:7]
                vel_pos_error = -env.joint_velocities()[:7]
                desired_torque = (np.multiply(np.array(position_error), np.array(kp))
                                  + np.multiply(vel_pos_error, kd))
                # target_pos = curr_pos # update target position for rendering
            else: # controller 2, pre-move phase
                F, error = compute_ts_force(curr_pos, curr_ori, target_pos, original_ori, curr_vel, curr_omg, target_vel)
                desired_torque = np.dot(env.get_ee_jacobian("ee_joint").T, F).flatten().tolist()

            # Return desired torques plus gravity compensations
            MassMatrix = joint_controller.dynamics.MassMatrix()
            desired_torque = MassMatrix.dot(desired_torque) + joint_controller.dynamics.gravityforces(robot.joint_pos)
            robot.set_joint_torque(desired_torque)

            env.sim.step()

            elapsed_c = time.time() - now_c
            sleep_time_c = (1. / ctrl_rate) - elapsed_c
            if sleep_time_c > 0.0:
                print(f"sleep_time_c:{sleep_time_c}")
                time.sleep(sleep_time_c)
            # time.sleep(0.002)

            step += 1

def go_to_initial_pos():
    global env, robot, joint_controller
    # initial_pos = np.array([0.0, 0.85, 0.0, -1.7, -1.62, 1.6, 0.0])
    initial_pos = np.array([0.0, 0.85, -0.2, -1.7, -1.62, 1.6, 1.0])
    initial_pos2 = np.array([0.0, 0.85, -0.3, -1.7, -1.62, 1.95, 2.0])


    env.set_initial_pos(initial_pos)
    env.render()

    while True:
        jtor, su_flag = joint_controller.impedance_controller_joint()
        robot.set_joint_torque(jtor)
        env.sim.step()
        env.render()

        if su_flag:
            break
        sleep(0.005)

if __name__ == "__main__":
    model = load_model_from_path("./description/Franka/dynamic_franka_without_gripper_with_object.xml")
    sim = MjSim(model=model)
    env = environments.Base_env(sim)
    robot = robots.Franka_robot(sim)
    joint_controller = JointPositionImpedanceGenerator(1000, 4.1, robot)
    target1 = np.array([0.0, 0.9, 0.0, -1.7, -1.62, 1.6, 0.0])  # target 1
    target2 = np.array([0.0, 0.9, -0.4, -1.7, -1.62, 1.95, 2.0])  # target 2
    joint_controller.set_target(target2)

    go_to_initial_pos()
    print("initial position")

    rospy.init_node('talker', anonymous=True)

    pub_d = rospy.Publisher('distance', Float32, queue_size=10)  # the distance between object and robot's end effector
    pub_f = rospy.Publisher('contact_force', Float32, queue_size=10)  # the contact force

    pub_vel = rospy.Publisher('object_velocity', Float32, queue_size=10)
    pub_robot_vel = rospy.Publisher('robot_velocity', Float32, queue_size=10)

    force_norm = 0
    env.sim.data.xfrc_applied[env.sim.model.body_name2id("object"), :] = np.array([0, 40, 0, 0, 0, 0])
    print("applied force!")
    for i in range(2000):
        env.sim.step()
        env.render()
        # vel = math.sqrt(env.joint_velocities()[7] ** 2 + env.joint_velocities()[8] ** 2)
        vel = env.joint_velocities()[7] 
        pub_vel.publish(vel)
        pub_f.publish(force_norm)
        # print("applied force!")
        joint_pos = env.get_site_pos("ee_joint")
        object_pos = env.get_site_pos("obj_contact")
        dis = np.sqrt(np.sum(np.square(joint_pos - object_pos)))
        if dis <= 1:
            env.sim.data.xfrc_applied[env.sim.model.body_name2id("object"), :] = np.array([0, 0, 0, 0, 0, 0])
        if dis < 0.3:
            break
        sleep(0.001)

    if dis <= 0.3:
        run_controller = True
    else:
        run_controller = False

    ctrl_rate = 1 / env.sim.model.opt.timestep
    render_rate = 100
    curr_ee, original_ori = env.get_ee_pose()  # end effector's pose
    curr_vel_ee, curr_omg_ee = env.get_ee_velocity()  # end effector's velocity
    target_pos = curr_ee.copy()
    target_y_vel = np.linspace(curr_vel_ee[1], curr_vel_ee[1] + 0.4, 100).tolist()
    y_target_vel = curr_vel_ee[1]
    target_vel = curr_vel_ee.copy()

    ctrl_thread = threading.Thread(target=impedance_control_integration, args=[ctrl_rate])  # 传递的是固定参数，这两个是交错执行的，先执行上面再执行下面
    ctrl_thread.start()

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