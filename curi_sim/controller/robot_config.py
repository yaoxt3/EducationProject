from . import tool as T
import numpy as np

class Robot(object):
    def __init__(self, sim, screw=None, initial_pose=None):
        '''
        :param sim (MjSim): Mujoco sim
        :param screw (np.array): The joint screw axes in the space frame
        '''
        self.sim = sim
        self.screw = screw
        self.initial_pose = initial_pose
        self._ref_joint_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.robot_model.actuators
        ]


    @property
    def joint_pos(self):
        '''
        :return qpos (np.array): joint position
        '''
        return self.sim.data.qpos[0:7]

    @property
    def joint_vel(self):
        """
        :return joint angular velocities (np.array)
        """
        return self.sim.data.qvel[0:7]

    @property
    def joint_acc(self):
        """
        :return joint angular velocities (np.array)
        """
        return self.sim.data.qacc[0:7]

    # @property
    # def torque_limits(self):
    #     """
    #     Torque lower/upper limits per dimension.
    #     Returns:
    #         2-tuple:
    #             - (np.array) minimum (low) torque values
    #             - (np.array) maximum (high) torque values
    #     """
    #     # Torque limit values pulled from relevant robot.xml file
    #     low = self.sim.model.actuator_ctrlrange[self._ref_joint_actuator_indexes, 0]
    #     high = self.sim.model.actuator_ctrlrange[self._ref_joint_actuator_indexes, 1]
    #
    #     return low, high

    def eef_pose_in_base(self):
        '''
        :return pose (np.array): The pose matrix (position and orientation) of panda0_link7
        '''
        return self.pose_in_base_from_name('panda0_link7')

    def set_joint_torque(self, jfrc):
        # for i in range(len(jfrc)):
        #     if jfrc[i] <-80:
        #         jfrc[i] = -80
        #     if jfrc[i] >80:
        #         jfrc[i] = 80
        self.sim.data.qfrc_applied[0:7] = jfrc
        self.sim.forward()

    def set_joint_pos(self, jpos):
        '''
        :param jpos (np.array): Set joints position of the robot
        '''
        self.sim.data.qpos[0:7] = jpos
        self.sim.forward()

    def get_link_M(self):
        T1 = np.array([[1, 0, 0, 0.003875], [0, 1, 0, 0.002081], [0, 0, 1, 0], [0, 0, 0, 1]])
        T2 = np.array([[1, 0, 0, -3.141e-03], [0, 1, 0, -2.872e-02], [0, 0, 1, 3.495e-03], [0, 0, 0, 1]])
        T3 = np.array([[1, 0, 0, 2.7518e-02], [0, 1, 0, 3.9252e-02], [0, 0, 1, -6.6502e-02], [0, 0, 0, 1]])
        T4 = np.array([[1, 0, 0, -5.317e-02], [0, 1, 0, 1.04419e-01], [0, 0, 1, 2.7454e-02], [0, 0, 0, 1]])
        T5 = np.array([[1, 0, 0, -1.1953e-02], [0, 1, 0, 4.1065e-02], [0, 0, 1, -3.8437e-02], [0, 0, 0, 1]])
        T6 = np.array([[1, 0, 0, 6.0149e-02], [0, 1, 0, -1.4117e-02], [0, 0, 1, -1.0517e-02], [0, 0, 0, 1]])
        T7 = np.array([[1, 0, 0, 1.0517e-02], [0, 1, 0, -4.252e-03], [0, 0, 1, 6.1597e-02], [0, 0, 0, 1]])
        M01 = self.pose_in_base_from_name('panda0_link1').dot(T1)   #0l lg  0g
        M02 = self.pose_in_base_from_name('panda0_link2').dot(T2)
        M03 = self.pose_in_base_from_name('panda0_link3').dot(T3)
        M04 = self.pose_in_base_from_name('panda0_link4').dot(T4)
        M05 = self.pose_in_base_from_name('panda0_link5').dot(T5)
        M06 = self.pose_in_base_from_name('panda0_link6').dot(T6)
        M07 = self.pose_in_base_from_name('panda0_link7').dot(T7)

        M10= self.pose_inv(M01)
        M20 = self.pose_inv(M02)
        M30 = self.pose_inv(M03)
        M40 = self.pose_inv(M04)
        M50 = self.pose_inv(M05)
        M60 = self.pose_inv(M06)

        self.M01 = M01
        self.M12 = M10.dot(M02)
        self.M23 = M20.dot(M03)
        self.M34 = M30.dot(M04)
        self.M45 = M40.dot(M05)
        self.M56 = M50.dot(M06)
        self.M67 = M60.dot(M07)
        self.M78 = self.M67 #external force related to link 7

        return np.array([self.M01, self.M12, self.M23, self.M34, self.M45, self.M56, self.M67, self.M78])



    def pose_inv(self, pose):
        """
        Computes the inverse of a homogeneous matrix corresponding to the pose of some
        frame B in frame A. The inverse is the pose of frame A in frame B.
        Args:
            pose (np.array): 4x4 matrix for the pose to inverse
        Returns:
            np.array: 4x4 matrix for the inverse pose
        """

        # Note, the inverse of a pose matrix is the following
        # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

        # Intuitively, this makes sense.
        # The original pose matrix translates by t, then rotates by R.
        # We just invert the rotation by applying R-1 = R.T, and also translate back.
        # Since we apply translation first before rotation, we need to translate by
        # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
        # R-1 to align the axis again.

        pose_inv = np.zeros((4, 4))
        pose_inv[:3, :3] = pose[:3, :3].T
        pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
        pose_inv[3, 3] = 1.0
        return pose_inv



    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
        Args:
            name (str): Name of body in sim to grab pose
        Returns:
            np.array: (4,4) array corresponding to the pose of @name in the base frame
        """
        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos('panda0_link0')
        base_rot_in_world = self.sim.data.get_body_xmat('panda0_link0').reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

class Franka(Robot):
    def __init__(self,sim):
        '''
        :param sim (MjSim): Mujoco sim
        '''
        Slist = np.array([[0, 0, 1, 0, 0, 0],
                          [0, 1, 0, -0.333, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, -1, 0, 0.649, 0, -0.0825],
                          [0, 0, 1, 0, 0, 0],
                          [0, -1, 0, 1.033, 0, 0],
                          [0, 0, -1, 0, 0.088, 0]]).T  #The joint screw axes in the space frame
        self.sim = sim
        self.screw = Slist
        self.initial_pose = np.array([[1, 0, 0, 0.088],
              [0, -1, 0, 0],
              [0, 0, -1, 1.033],
              [0, 0, 0, 1]])
        # G1 = np.diag([0.17, 0.17, 0.042, 2.70, 2.70, 2.70])
        # G2 = np.diag([0.16, 0.15, 0.053, 2.73, 2.73, 2.73])
        # G3 = np.diag([0.87, 0.852, 0.365, 2.0, 2.0, 2.0])
        # G4 = np.diag([0.900, 0.883, 0.377, 2.08, 2.08, 2.08])
        # G5 = np.diag([0.261, 0.252, 0.0534, 3.0, 3.0, 3.0])
        # G6 = np.diag([0.401 ,0.328 ,0.207, 1.32, 1.32, 1.32])
        # G7 = np.diag([0.305, 0.253, 0.133, 0.2, 0.2, 0.2])

        # original version
        # G1 = np.diag([0.70337, 0.70661, 0.009117, 4.970684, 4.970684, 4.970684])
        # G2 = np.diag([0.007962, 0.02811, 0.025995, 0.646926, 0.646926, 0.646926])
        # G3 = np.diag([0.037242, 0.036155, 0.01083, 3.228604, 3.228604, 3.228604])
        # G4 = np.diag([0.025853, 0.019552, 0.028323, 3.587895, 3.587895, 3.587895])
        # G5 = np.diag([0.035549, 0.029474, 0.008627, 1.225946, 1.225946, 1.225946])
        # G6 = np.diag([0.001964, 0.004354, 0.005433, 1.666555, 1.666555, 1.666555])
        # G7 = np.diag([0.012516, 0.010027, 0.004815, 7.35522e-01, 7.35522e-01, 7.35522e-01])

        # official version
        G1 = np.diag([7.0337e-01, 7.0661e-01, 9.1170e-03, -1.3900e-04, 6.7720e-03, 1.9169e-02])
        G2 = np.diag([7.9620e-03, 2.8110e-02, 2.5995e-02, -3.9250e-03, 1.0254e-02, 7.0400e-04])
        G3 = np.diag([3.7242e-02, 3.6155e-02, 1.0830e-02, -4.7610e-03, -1.1396e-02, -1.2805e-02])
        G4 = np.diag([2.5853e-02, 1.9552e-02, 2.8323e-02, 7.7960e-03, -1.3320e-03, 8.6410e-03])
        G5 = np.diag([3.5549e-02, 2.9474e-02, 8.6270e-03, -2.1170e-03, -4.0370e-03, 2.2900e-04])
        G6 = np.diag([1.9640e-03, 4.3540e-03, 5.4330e-03, 1.0900e-04, -1.1580e-03, 3.4100e-04])
        G7 = np.diag([1.2516e-02, 1.0027e-02, 4.8150e-03, -4.2800e-04, -1.1960e-03, -7.4100e-04])
        self.inertia_list = np.array([G1, G2, G3, G4, G5, G6, G7])




#
#
#
# class Curi(Robot):
#
#     def joint_pos(self):
#         pass
#
#     def eef_pose(self):
#         pass
#
#     def set_joint_torque(self):
#         pass
#
#     def set_joint_pos(self):
#         pass



