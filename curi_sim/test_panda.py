import robots
import environments
from mujoco_py import load_model_from_path, MjSim
import numpy as np
from time import sleep
from controller import Dynamics, JointPositionImpedanceGenerator

model = load_model_from_path("./description/Franka/dynamic_franka_without_gripper.xml")
sim = MjSim(model=model)  # generate sim by model xml later
env = environments.Base_env(sim)
robot = robots.Franka_robot(sim)
dynamics = Dynamics(robot)
joint_controller = JointPositionImpedanceGenerator(700, 4.1, robot)
target = np.array([-0.93, -0.95, 0.803, -2.41, -0.13, 2.124, 0.935])



# while(1):
#     #print(env.sim.data.qacc)
#     #jtor = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#     theta = robot.joint_pos
#     jtor2 = dynamics.gravityforces(theta)
#     jtor = robot.torque_compensation
#     # print(jtor2)
#     # print(jtor)
#
#     robot.set_joint_torque(jtor2)
#     env.sim.step()
#     env.render()



while(1):
    #print(env.sim.data.qacc)
    #jtor = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    theta = robot.joint_pos

    # target = np.array([-0.93, -0.95, 0.803, -2.41, -0.13, 2.124, 0.935])
    # joint_controller.set_target(target)
    # while True:
    #     jtor,su_flag = joint_controller.impedance_controller_joint()
    #     robot.set_joint_torque(jtor)
    #     env.sim.step()
    #     env.render()
    #     if su_flag:
    #         break
    #
    # target = np.array([-0., -0., 0., -0, -0, 0, 0])
    # joint_controller.set_target(target)
    # while True:
    #     jtor, su_flag = joint_controller.impedance_controller_joint()
    #     robot.set_joint_torque(jtor)
    #     #print(jtor)
    #     env.sim.step()
    #     env.render()
    #     if su_flag:
    #         break

    # print(f'xfrc:{env.sim.data.xfrc_applied[env.sim.model.body_name2id("panda0_link7"),:]}')
    # target = np.array([-0.93, -0.95, 0.803, -2.41, -0.13, 2.124, 0.935])
    target = np.array([0.0, 0.74, 0.0, -1.7, -1.62, 1.53, 0.0])
    joint_controller.set_target(target)
    flag = 0
    while True:
        # env.sim.data.xfrc_applied[env.sim.model.body_name2id("panda0_link7"), :] = np.array([0, 0, 0, 0, 0, 0])
        jtor, su_flag = joint_controller.impedance_controller_joint()
        robot.set_joint_torque(jtor)
        env.sim.step()
        env.render()
        print(f"su_flag: {su_flag}")
        print(f'xfrc:{env.sim.data.xfrc_applied[env.sim.model.body_name2id("panda0_link7"), :]}')

        if su_flag and flag <= 0:
            env.sim.data.xfrc_applied[env.sim.model.body_name2id("panda0_link7"), :] = np.array([10, 0, 0, 0, 0, 0])
            for i in range(100):
                env.sim.step()
                env.render()
                sleep(0.05)
                print("applied force!")
            flag = 1
            env.sim.data.xfrc_applied[env.sim.model.body_name2id("panda0_link7"), :] = np.array([0, 0, 0, 0, 0, 0])
            # break


