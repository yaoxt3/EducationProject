import math
import robots
import environments
from mujoco_py import load_model_from_path, MjSim
import mujoco_py as mjp
import numpy as np
from time import sleep
from controller import Dynamics, JointPositionImpedanceGenerator
import rospy
from std_msgs.msg import String, Float32

model = load_model_from_path("./description/Franka/dynamic_franka_without_gripper_with_object.xml")
# model = load_model_from_path("./description/Franka/dynamic_franka_without_gripper.xml")
sim = MjSim(model=model)  # generate sim by model xml later
env = environments.Base_env(sim)
# robot = robots.Franka_robot(sim, np.array([0.0, 0.5, 0.0, -1.7, -1.62, 1.53, 0.0]))
robot = robots.Franka_robot(sim)
# robot.set_initial_pos(np.array([0.0, 0.74, 0.0, -1.7, -1.62, 1.53, 0.0]))
dynamics = Dynamics(robot)
joint_controller = JointPositionImpedanceGenerator(1000, 4.1, robot)
target = np.array([-0.93, -0.95, 0.803, -2.41, -0.13, 2.124, 0.935])

def mujocoFunc():
    # print(f'xfrc:{env.sim.data.xfrc_applied[env.sim.model.body_name2id("panda0_link7"),:]}')
    # target = np.array([-0.93, -0.95, 0.803, -2.41, -0.13, 2.124, 0.935])
    # target1 = np.array([0.0, 0.74, 0.0, -1.7, -1.62, 1.6, 0.0])  # target 1
    # ros init
    rospy.init_node('talker', anonymous=True)

    pub_d = rospy.Publisher('distance', Float32, queue_size=10)  # the distance between object and robot's end effector
    pub_f = rospy.Publisher('contact_force', Float32, queue_size=10)  # the contact force

    # publish the velocity of object
    pub_vel = rospy.Publisher('object_velocity', Float32, queue_size=10)

    theta = robot.joint_pos
    target1 = np.array([0.0, 0.9, 0.0, -1.7, -1.62, 1.6, 0.0])  # target 1
    joint_controller.set_target(target1)
    flag = 0
    step = 0
    vel = 0
    while True:
        # env.sim.data.xfrc_applied[env.sim.model.body_name2id("panda0_link7"), :] = np.array([0, 0, 0, 0, 0, 0])
        jtor, su_flag = joint_controller.impedance_controller_joint()
        robot.set_joint_torque(jtor)
        env.sim.step()
        env.render()
        joint_pos = env.get_site_pos("joint_contact")
        object_pos = env.get_site_pos("obj_contact")
        dis = np.sqrt(np.sum(np.square(joint_pos)))
        print(f'distance: {dis}')
        pub_d.publish(dis)
        force_norm = 0
        print(f'Joint velocity:{env.sim.data.qvel}')
        vel = math.sqrt(env.sim.data.qvel[7]**2+env.sim.data.qvel[8]**2)
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
            if sim.model.geom_id2name(contact.geom1)=="link7_col" or sim.model.geom_id2name(contact.geom2)=="link7_col":
                force_norm = np.sqrt(np.sum(np.square(c_array[0:3])))
                print("Force Norm", force_norm)
            print('c_array', c_array)
        pub_f.publish(force_norm)
        step += 1

        # when the robot reach to desired position, apply force to the object
        if su_flag and flag <= 0:
            env.sim.data.xfrc_applied[env.sim.model.body_name2id("object"), :] = np.array([0, 80, 0, 0, 0, 0])
            for i in range(70):
                env.sim.step()
                env.render()
                sleep(0.05)
                vel = math.sqrt(env.sim.data.qvel[7] ** 2 + env.sim.data.qvel[8] ** 2)
                pub_vel.publish(vel)
                print("applied force!")
            flag = 1
            env.sim.data.xfrc_applied[env.sim.model.body_name2id("object"), :] = np.array([0, 0, 0, 0, 0, 0])
            # break
        sleep(0.01)


if __name__ == "__main__":
    mujocoFunc()