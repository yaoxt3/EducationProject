from mujoco_py import MjSim, MjViewer
import mujoco_py
import curi_sim
import numpy as np
import quaternion

class Base_env(object):
    def __init__(self, sim):
        self.sim = sim
        self.view = mujoco_py.MjViewer(self.sim)

    def render(self, *args):
        self.view.render()

    def reset(self, *args):
        self.sim.reset()

    def step(self, *args):
        self.sim.step()

    def get_site_pos(self, siteName):
        id = self.sim.model.site_names.index(siteName)
        return self.sim.data.site_xpos[id].copy()

    def get_site_id(self, siteName):
        return self.sim.model.site_names.index(siteName)

    def get_ee_jacobian(self, siteName):
        id = self.get_site_id(siteName)
        jacp = self.sim.data.site_jacp[id, :].reshape(3, -1)
        jacr = self.sim.data.site_jacr[id, :].reshape(3, -1)
        joint_indices = [0, 1, 2, 3, 4, 5, 6]
        return np.vstack([jacp[:, joint_indices], jacr[:, joint_indices]])

    def get_site_pose(self, site_id):
        return self.sim.data.site_xpos[site_id].copy(), quaternion.as_float_array(quaternion.from_rotation_matrix(self.sim.data.site_xmat[site_id].copy().reshape(3, 3)))

    def get_site_velocity(self, site_id):
        return self.sim.data.site_xvelp[site_id].copy(), self.sim.data.site_xvelr[site_id].copy()

    def get_ee_pose(self):
        ee_site_id = self.get_site_id("ee_joint")
        return self.get_site_pose(ee_site_id)

    def get_ee_velocity(self):
        ee_site_id = self.get_site_id("ee_joint")
        return self.get_site_velocity(ee_site_id)

    def joint_position(self):
        return self.sim.data.qpos.copy()

    def joint_velocities(self):
        return self.sim.data.qvel.copy()

    def set_initial_pos(self, initial_pos):
        self.sim.data.qpos[0:7] = initial_pos
        self.sim.forward()
