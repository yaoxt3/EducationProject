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



def mujocoJointFunc():
    global env, robot, joint_controller
    rospy.init_node('talker', anonymous=True)

    pub_d = rospy.Publisher('distance', Float32, queue_size=10)  # the distance between object and robot's end effector
    pub_f = rospy.Publisher('contact_force', Float32, queue_size=10)  # the contact force

    # publish the velocity of object
    pub_vel = rospy.Publisher('object_velocity', Float32, queue_size=10)

    target1 = np.array([0.0, 0.9, 0.0, -1.7, -1.62, 1.6, 0.0])  # target 1
    joint_controller.set_target(target1)
    flag = 0
    step = 0
    vel = 0

    initial_pos = np.array([0.0, 0.8, 0.0, -1.7, -1.62, 1.6, 0.0])

    env.set_initial_pos(initial_pos)
    env.render()

    while True:
        # env.sim.data.xfrc_applied[env.sim.model.body_name2id("panda0_link7"), :] = np.array([0, 0, 0, 0, 0, 0])
        jtor, su_flag = joint_controller.impedance_controller_joint()
        robot.set_joint_torque(jtor)
        env.sim.step()
        env.render()

        # vel = math.sqrt(env.joint_velocities()[7] ** 2 + env.joint_velocities()[8] ** 2)
        # print(f"env vel: {vel}")
        #jacobian of end effector:
        jacobian = env.get_ee_jacobian("ee_joint")
        # print(f"step:{step}, jacobian: {jacobian}")

        joint_pos = env.get_site_pos("ee_joint")
        object_pos = env.get_site_pos("obj_contact")
        dis = np.sqrt(np.sum(np.square(joint_pos-object_pos)))

        pub_d.publish(dis)
        force_norm = 0
        vel = math.sqrt(env.sim.data.qvel[7]**2+env.sim.data.qvel[8]**2) # object's velocity
        pub_vel.publish(vel)
        for i in range(sim.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = sim.data.contact[i]
            print('contact', i)
            print('distance', contact.dist)
            print('geom1', contact.geom1, sim.model.geom_id2name(contact.geom1))
            print('geom2', contact.geom2, sim.model.geom_id2name(contact.geom2))
            # There's more stuff in the data structure
            # See the mujoco documentation for more info!
            geom2_body = sim.model.geom_bodyid[sim.data.contact[i].geom2]
            # Use internal functions to read out mj_contactFor
            c_array = np.zeros(6, dtype=np.float64)
            mjp.functions.mj_contactForce(sim.model, sim.data, i, c_array)
            if sim.model.geom_id2name(contact.geom1) == "link7_col" or \
                    sim.model.geom_id2name(contact.geom2) == "link7_col":
                force_norm = np.sqrt(np.sum(np.square(c_array[0:3])))
                print("Force Norm", force_norm)
            print('c_array', c_array)
        pub_f.publish(force_norm)
        step += 1
        # when the robot reach to desired position, apply force to the object
        if su_flag and flag <= 0:
            env.sim.data.xfrc_applied[env.sim.model.body_name2id("object"), :] = np.array([0, 10, 0, 0, 0, 0])
            for i in range(70):
                env.sim.step()
                env.render()
                sleep(0.05)
                vel = math.sqrt(env.sim.data.qvel[7] ** 2 + env.sim.data.qvel[8] ** 2)
                pub_vel.publish(vel)
                pub_f.publish(force_norm)
                print("applied force!")
            flag = 1
            env.sim.data.xfrc_applied[env.sim.model.body_name2id("object"), :] = np.array([0, 0, 0, 0, 0, 0])
            # break
        
        sleep(0.01)

# --------- Modify as required ------------
# Task-space controller parameters
# stiffness gains
P_pos = 1500.
P_ori = 600.
# damping gains
D_pos = 2.*np.sqrt(P_pos)
D_ori = 2.
# -----------------------------------------

# impedance control based on position
# phase 1: robot pre-move
# phase 2: object and robot contact
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

def impedance_control_integration(ctrl_rate):
    print("impedance_control_integration")
    global env, target_pos, target_vel, original_ori
    count = 0
    threshold = 0.0000005

    target_pos = curr_ee.copy()
    target_vel = curr_vel_ee.copy()
    step = 1
    jpos_in_contact = []
    stop_flag = 0
    force_norm = 0

    while run_controller:
        error = 100.

        while error > threshold:
            now_c = time.time()
            curr_pos, curr_ori = env.get_ee_pose()
            curr_vel, curr_omg = env.get_ee_velocity() # get linear velocity and angular velocity

            target_vel[1] = y_target_vel
            delta_vel = y_target_vel * env.sim.model.opt.timestep * step*2

            while True:
                if stop_flag == 0:
                    target_pos[1] = curr_ee[1] + delta_vel
                    print(f"update target position")
                # elif stop_flag == 1:
                #     target_pos[1] = curr_ee[1] + force_norm
                break

            # print(f"target_pos: {target_pos}, target_vel:{target_vel}")

            joint_pos = env.get_site_pos("ee_joint")
            object_pos = env.get_site_pos("obj_contact")
            dis = np.sqrt(np.sum(np.square(joint_pos[1] - object_pos[1])))
            pub_d.publish(dis)

            # vel = math.sqrt(env.joint_velocities()[7] ** 2 + env.joint_velocities()[8] ** 2)
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
                    # print('c_array', c_array)

            print("Force Norm", force_norm)
            while force_norm > 0:
                count += 1
                break

            while True:
                if force_norm<0.1 and count > 0 and stop_flag==0:
                    print(f"stop position")
                    target_pos, target_ori = env.get_ee_pose()
                    target_pos[1] += 0.05
                    stop_flag = 1
                break

            F, error = compute_ts_force(curr_pos, curr_ori, target_pos, original_ori, curr_vel, curr_omg, target_vel)
            impedance_acc_des = np.dot(env.get_ee_jacobian("ee_joint").T, F).flatten().tolist()

            MassMatrix = joint_controller.dynamics.MassMatrix()
            # Return desired torques plus gravity compensations
            impedance_acc_des = MassMatrix.dot(impedance_acc_des) + joint_controller.dynamics.gravityforces(
                robot.joint_pos)
            robot.set_joint_torque(impedance_acc_des)

            # impedance_acc_des, flag = joint_controller.impedance_controller_joint()
            # robot.set_joint_torque(impedance_acc_des)
            
            print(f"target:{joint_controller.get_target()}")
            print(f"impedance_acc_des: {impedance_acc_des}")

            if error <= threshold:
                break

            env.sim.step()

            elapsed_c = time.time() - now_c
            # sleep_time_c = (1. / ctrl_rate) - elapsed_c
            # if sleep_time_c > 0.0:
            #     print(f"sleep_time_c:{sleep_time_c}")
            #     time.sleep(sleep_time_c)
            # time.sleep(0.0002)

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

    # publish the velocity of object
    pub_vel = rospy.Publisher('object_velocity', Float32, queue_size=10)

    ctrl_rate = 1 / env.sim.model.opt.timestep
    render_rate = 100
    curr_ee, original_ori = env.get_ee_pose() # end effector's pose
    curr_vel_ee, curr_omg_ee = env.get_ee_velocity() # end effector's velocity
    target_pos = curr_ee.copy()
    target_y_vel = np.linspace(curr_vel_ee[1], curr_vel_ee[1] + 0.4, 100).tolist()
    y_target_vel = curr_vel_ee[1]

    target_vel = curr_vel_ee

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
        if dis <= 1.:
            env.sim.data.xfrc_applied[env.sim.model.body_name2id("object"), :] = np.array([0, 0, 0, 0, 0, 0])
        if dis < 0.4:
            break
        sleep(0.001)
    print(f"dis:{dis}")
    print("cancel force")
    if dis <= 0.4:
        print("run_controller True")
        run_controller = True
    else:
        run_controller = False

    ctrl_thread = threading.Thread(target=impedance_control_integration, args=[ctrl_rate])  # 传递的是固定参数，这两个是交错执行的，先执行上面再执行下面
    ctrl_thread.start()

    now_r = time.time()
    i = 0

    while i < len(target_y_vel):

        y_target_vel = target_y_vel[i]
        print(f"y_target_vel:{y_target_vel}")
        robot_pos, robot_ori = env.get_ee_pose()
        elapsed_r = time.time() - now_r
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