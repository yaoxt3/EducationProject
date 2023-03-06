# ----------- Inverse Kinematics -----------
from dm_control import mujoco
from dm_control.mujoco.testing import assets
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import inverse_kinematics as ik
from dm_control.mujoco.testing import assets
import numpy as np
from time import sleep
import time

mjlib = mjbindings.mjlib

_ARM_XML = assets.get_contents('/home/yxt/Research/code/EducationProject/curi_sim/description/Franka/arm_for_IK.xml')
_SITE_NAME = 'ee_joint'
_JOINTS = ['panda0_joint1', 'panda0_joint2', 'panda0_joint3', 'panda0_joint4', 'panda0_joint5', 'panda0_joint6', 'panda0_joint7']
_TOL = 1.2e-8
_MAX_STEPS = 150
_MAX_RESETS = 10


class _ResetArm:
  def __init__(self, seed=None):
    self._rng = np.random.RandomState(seed)
    self._lower = None
    self._upper = None

  def _cache_bounds(self, physics):
    self._lower, self._upper = physics.named.model.jnt_range[_JOINTS].T
    limited = physics.named.model.jnt_limited[_JOINTS].astype(bool)
    # Positions for hinge joints without limits are sampled between 0 and 2pi
    self._lower[~limited] = 0
    self._upper[~limited] = 2 * np.pi

  def __call__(self, physics):
    if self._lower is None:
      self._cache_bounds(physics)
    # NB: This won't work for joints with > 1 DOF
    new_qpos = self._rng.uniform(self._lower, self._upper)
    physics.named.data.qpos[_JOINTS] = new_qpos
 
   
# set starting joint position
def ik_set_joint_pos(physics, joint_pos):
    physics.named.data.qpos[_JOINTS] = joint_pos


def inverse_kinematics(ik_target_pos, ik_target_quat, ik_current_pos):
    physics = mujoco.Physics.from_xml_string(_ARM_XML)
    count = 0
    physics2 = physics.copy(share_model=True)
    ik_set_joint_pos(physics2, ik_current_pos)
    step = 0
    curr_time = time.time()
    while True:
        result = ik.qpos_from_site_pose(
            physics=physics2,
            site_name=_SITE_NAME,
            target_pos=ik_target_pos,
            target_quat=ik_target_quat,
            joint_names=_JOINTS,
            tol=_TOL,
            max_steps=_MAX_STEPS,
            inplace=False,
        )
        print(f"step:{step}, result:{result}")
        if result.success:
            break
        elif count < _MAX_RESETS:
            ik_set_joint_pos(physics2, ik_current_pos)
            count += 1
        else:
            raise RuntimeError(
                'Failed to find a solution within %i attempts.' % _MAX_RESETS)
        step += 1
        
    print(f"ik time: {time.time() - curr_time}") 
    
    physics.data.qpos[:] = result.qpos
    mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)
    if ik_target_pos is not None:
      pos = physics.named.data.site_xpos[_SITE_NAME]
      np.testing.assert_array_almost_equal(pos, ik_target_pos)
    
    if ik_target_quat is not None:
      xmat = physics.named.data.site_xmat[_SITE_NAME]
      quat = np.empty_like(ik_target_quat)
      mjlib.mju_mat2Quat(quat, xmat)
      quat /= quat.ptp()  # Normalize xquat so that its max-min range is 1
      np.testing.assert_array_almost_equal(quat, ik_target_quat)
      
    return result.qpos