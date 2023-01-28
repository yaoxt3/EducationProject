from . import core
import numpy as np

class Dynamics(object):
    def __init__(self, robot):
        '''
        :param robot (robot_mujoco.Robot): instance of class Robot
        '''
        self.robot = robot
        self.Mlist = self.robot.get_link_M()

    def id(self,thetalist, dthetalist, ddthetalist,Ftip):
        Mlist = self.Mlist
        Glist = self.robot.inertia_list
        g = np.array([0, 0, -9.8])
        j_tor_target = core.InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, self.robot.screw)
        return j_tor_target

    def gravityforces(self, thetalist):
        Mlist = self.Mlist
        Glist = self.robot.inertia_list
        g = np.array([0, 0, -10])
        torque = core.GravityForces(thetalist, g, Mlist, Glist, self.robot.screw)
        return torque

    def Quadraticforces(self, thetalist, dthetalist):
        Mlist = self.Mlist
        Glist = self.robot.inertia_list
        torque = core.VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, self.robot.screw)
        return torque

    def MassMatrix(self):
        return core.MassMatrix(self.robot.joint_pos, self.Mlist, self.robot.inertia_list, self.robot.screw)

    def torque_controller(self):
        # torques = pos_err * kp + vel_err * kd
        mass_matrix=core.MassMatrix(self.robot.joint_pos, self.Mlist, self.robot.inertia_list, self.robot.screw)
        desired_qpos = np.zeros(7)
       # desired_qpos[1]=0.2
        position_error = desired_qpos - self.robot.joint_pos
        vel_pos_error = -self.robot.joint_vel
        desired_torque = (np.multiply(np.array(position_error), np.array(100000))
                          + np.multiply(vel_pos_error, 10))
        # Return desired torques plus gravity compensations
        #print(np.multiply(np.array(position_error), np.array(100)))
        torques = np.dot(mass_matrix, desired_torque) + self.gravityforces(self.robot.joint_pos) + self.Quadraticforces(self.robot.joint_pos,self.robot.joint_vel)
        return torques


