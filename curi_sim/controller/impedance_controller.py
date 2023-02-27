
import numpy as np
import numpy
import math
from . import core, dynamics


_EPS = numpy.finfo(float).eps * 4.0


class JointPositionImpedanceGenerator(object):
    def __init__(self, kp, kd, robot):
        self.kp = kp
        self.kd = kd
        self.robot = robot
        self.dynamics = dynamics.Dynamics(robot)
        self.min_diff = 0.05

    def update_goal_pos(self, pose_now, target):
        goal_pose = np.zeros((4, 4))
        goal_pose[3,3]=1
        # pose_now = self.robot.pose_in_base_from_name('panda0_link7')
        print(f"pose_now:{pose_now}")
        print(f"target:{target}")
        goal_pose[:3, :3] = pose_now[:3, :3].dot(core.MatrixExp3(0.05 * core.MatrixLog3(core.RotInv(pose_now[:3, :3]).dot(target[:3, :3]))))
        #print(goal_pose[:3, 3])
        goal_pose[:3, 3] = target[:3, 3] * 0.05 + (1 - 0.05) * pose_now[:3, 3]
        return goal_pose

    def impedance_controller_wrench(self, target):
        """
        Calculates the torques required to reach the desired setpoint
        Returns:
             np.array: Command torques
        """
        su_flag = False
        pose_now = self.robot.eef_pose_in_base()

        # Update state
        goal_pose = self.update_goal_pos(pose_now, target)
        error = np.zeros((4, 4))
        error[3, 3] = 1
        error[:3, :3] = core.RotInv(goal_pose[:3, :3]).dot(pose_now[:3, :3])
        error[:3, 3] = pose_now[:3, 3] - goal_pose[:3, 3]
        se3 = core.MatrixLog6(error)
        twist = core.se3ToVec(se3)
        wrench = np.zeros((6))  #twist

        wrench[:3] = twist[3:]
        wrench[3:] = twist[:3]  #wrench
        print('wrench', wrench)
        J_S = core.JacobianSpace(self.robot.screw, self.robot.joint_pos)
        wrench = -self.kp.dot(wrench) - self.kd.dot(J_S.dot(self.robot.joint_vel))
       # MassMatrix = self.dynamics.MassMatrix()
        desired_torque = J_S.T.dot(wrench) + self.dynamics.gravityforces(self.robot.joint_pos) + self.dynamics.Quadraticforces(self.robot.joint_pos,self.robot.joint_vel)
        
        return desired_torque

    def set_target(self,target):
        self.total_steps = 2000
        self.step = 0
        self.target = target

    def get_target(self):
        return self.target

    def impedance_controller_joint(self):
        # This is a normal interpolation
        su_flag = False
        dx = (self.target - self.robot.joint_pos) * 0.01

        # print('self.diff', self.target - self.robot.joint_pos)
        # print('self.target', self.target)
        # print('self.joint_pos', self.robot.joint_pos)
        desired_qpos = self.robot.joint_pos + dx

        # torques = pos_err * kp + vel_err * kd
        position_error = desired_qpos - self.robot.joint_pos
        vel_pos_error = -self.robot.joint_vel
        desired_torque = (np.multiply(np.array(position_error), np.array(self.kp))
                          + np.multiply(vel_pos_error, self.kd))
        MassMatrix = self.dynamics.MassMatrix()
        # Return desired torques plus gravity compensations
        desired_torque = MassMatrix.dot(desired_torque) + self.dynamics.gravityforces(self.robot.joint_pos)  #self.dynamics.gravityforces(self.robot.joint_pos) #+ self.dynamics.Quadraticforces(self.robot.joint_pos,self.robot.joint_vel)
        
        # print(np.absolute(self.target - self.robot.joint_pos) < self.min_diff)
        # print('\n')
        #print(self.robot.torque_compensation)  #
        #print(self.dynamics.gravityforces(self.robot.joint_pos))#
        # print('\n')
        if (np.absolute(self.target - self.robot.joint_pos) < self.min_diff).all():
            # print((self.target - self.robot.joint_pos))
            su_flag = True

        return desired_torque, su_flag

class CuriJointPositionImpedanceGenerator(object):
    def __init__(self,kp,kd,robot):
        self.kp = kp
        self.kd = kd
        self.robot = robot
        self.dynamics = dynamics.Dynamics(robot)

    def update_goal_pos(self,pose_now,target):
        goal_pose = np.zeros((4, 4))
        goal_pose[3,3]=1
        # pose_now = self.robot.pose_in_base_from_name('panda0_link7')
        goal_pose[:3, :3] = pose_now[:3, :3].dot(core.MatrixExp3(0.05 * core.MatrixLog3(core.RotInv(pose_now[:3, :3]).dot(target[:3, :3]))))
        #print(goal_pose[:3, 3])
        goal_pose[:3, 3] = target[:3, 3] * 0.05 + (1 - 0.05) * pose_now[:3, 3]
        return goal_pose


    def set_target(self,target):
        self.total_steps = 2000
        self.step = 0
        self.target = target

    def impedance_controller_joint(self):
        # This is a normal interpolation
        su_flag = False
        dx = (self.target - self.robot.r_arm_joint_pos) *0.01
        print('self.step',self.target - self.robot.r_arm_joint_pos)
        desired_qpos = self.robot.r_arm_joint_pos + dx

        # torques = pos_err * kp + vel_err * kd
        position_error = desired_qpos - self.robot.r_arm_joint_pos
        vel_pos_error = -self.robot.r_arm_joint_vel
        desired_torque = (np.multiply(np.array(position_error), np.array(self.kp))
                          + np.multiply(vel_pos_error, self.kd))
        #MassMatrix = self.dynamics.MassMatrix()
        # Return desired torques plus gravity compensations
        desired_torque =  desired_torque + self.dynamics.gravityforces(self.robot.r_arm_joint_pos)#self.dynamics.gravityforces(self.robot.joint_pos) #+ self.dynamics.Quadraticforces(self.robot.joint_pos,self.robot.joint_vel)
        print(np.absolute(self.target - self.robot.r_arm_joint_pos) < 0.1)
        print('\n')
        #print(self.robot.torque_compensation)  #
        #print(self.dynamics.gravityforces(self.robot.joint_pos))#
        print('\n')

        if (np.absolute(self.target - self.robot.r_arm_joint_pos) < 0.1).all():
            print((self.target - self.robot.r_arm_joint_pos))
            su_flag = True

        return desired_torque,su_flag










