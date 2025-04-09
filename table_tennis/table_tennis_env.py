import os

import numpy as np

import gymnasium as gym
import mujoco
from gym import spaces
from mujoco import MjModel, MjData, Renderer
from mujoco import mj_name2id, mjtObj  # type: ignore

from scipy.spatial.transform import Rotation as R



def reset_ball(mj_model, mj_data):
    print("Resetting ball")
    target_ball_id = mj_name2id(mj_model, mjtObj.mjOBJ_BODY, "target_ball")
    body_jnt_addr = mj_model.body_jntadr[target_ball_id]
    assert body_jnt_addr >= 0, "Body joint address not found"
    qposadr = mj_model.jnt_qposadr[body_jnt_addr]
    mj_data.qpos[qposadr: qposadr + 3] = np.array([0, 0, 1.5])
    mj_data.qvel[qposadr: qposadr + 3] = np.array([2.0, 0, 0])

def get_geom_name(model, geom_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) if geom_id != -1 else None

def bat_hit_ball(model, data):
    """Returns True if bat1 hit the ball this step"""
    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = get_geom_name(model, contact.geom1)
        g2 = get_geom_name(model, contact.geom2)
        if 'bat1' in [g1, g2] and 'target_ball_contact' in [g1, g2]:
            return True
    return False


def ball_bounce_sides(model, data):
    """Returns a list of sides where the ball touched the table this step"""
    sides = []
    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = get_geom_name(model, contact.geom1)
        g2 = get_geom_name(model, contact.geom2)
        if 'target_ball_contact' in [g1, g2] and 'table_tennis_table' in [g1, g2]:
            ball_pos = data.site_xpos[model.site('target_ball')]
            if ball_pos[0] < 0:
                sides.append('opponent')
            else:
                sides.append('bat1')
    return sides


class TableTennisEnv(gym.Env):
    def __init__(self, model_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets/table_tennis_env.xml"),
                 max_step=800):
        super().__init__()

        # Load model and data
        model_path = model_path
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.object_names = ["target_ball", "bat1"]
        self.object_ids = []
        for object_name in self.object_names:
            object_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            assert object_id >= 0, f"Object {object_name} not found in model"
            self.object_ids.append(object_id)
        self.bat_id = self.object_ids[1]

        self.body_jnt_addr = self.mj_model.body_jntadr[self.bat_id]
        # assert self.body_jnt_addr >= 0, f"Joint not found for bat1{}"
        self.dofadr = self.mj_model.jnt_dofadr[self.body_jnt_addr]

        # the controller gains
        self.pos_gain = 200.0  # Position tracking gain
        self.rot_gain = 0.5  # Rotation tracking gain
        self.max_vel = 100.0  # Maximum linear velocity
        self.max_angvel = 40.0  # Maximum angular velocity

        # camera id
        self.cam_names = ["left_cam", "right_cam"]
        self.cam_ids = [self.mj_model.camera(name).id for name in self.cam_names]
        self.cam_renderer = Renderer(self.mj_model, height=224, width=224)

        # Renderer for visual output
        self.renderer = Renderer(self.mj_model)

        # contact state check
        self.hit_detected = False
        self.sides_hit = []
        self.success = False
        self.dones = False

        self.total_steps = 0
        self.max_steps = max_step

        # # Observation = qpos + qvel
        # n_obs = self.model.nq + self.model.nv
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        # )
        #
        # # Action = control inputs
        # self.action_space = spaces.Box(
        #     low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        # )

    def reset(self, seed=None, options=None):
        # mujoco.mj_resetData(self.mj_model, self.mj_data)

        reset_ball(self.mj_model, self.mj_data)
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.hit_detected = False
        self.sides_hit = []
        self.success = False
        self.dones = False
        self.total_steps = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):

        target_pos = action[:3]
        target_rot = R.from_euler("xyz", action[3:6], degrees=False)

        pos_error = target_pos - self.mj_data.xpos[self.bat_id]
        desired_vel = self.pos_gain * pos_error
        vel_norm = np.linalg.norm(desired_vel)
        if vel_norm > self.max_vel:
            desired_vel = desired_vel * (self.max_vel / vel_norm)
        self.mj_data.qvel[self.dofadr: self.dofadr + 3] = desired_vel
        # quat error
        # target_rot = R.from_quat(target_quat)
        current_quat = self.mj_data.xquat[self.bat_id].copy()
        current_rot = R.from_quat(current_quat[np.array([1, 2, 3, 0])])
        rot_error = current_rot.inv() * target_rot
        desired_angular_vel = self.rot_gain * rot_error.as_rotvec(degrees=True)
        ang_vel_norm = np.linalg.norm(desired_angular_vel)
        if ang_vel_norm > self.max_angvel:
            desired_angular_vel = desired_angular_vel * (self.max_angvel / ang_vel_norm)
        self.mj_data.qvel[self.dofadr + 3: self.dofadr + 6] = desired_angular_vel

        mujoco.mj_step(self.mj_model, self.mj_data)

        self.render()

        if not self.hit_detected and bat_hit_ball(self.mj_model, self.mj_data):
            self.hit_detected = True
            print("Bat1 hit the ball")

        if self.hit_detected:
            current_bounce_sides = ball_bounce_sides(self.mj_model, self.mj_data)
            for side in current_bounce_sides:
                if side not in self.sides_hit:
                    self.sides_hit.append(side)

            if "opponent" in self.sides_hit and "bat1" not in self.sides_hit:
                print("successfully returen the ball")
                self.success = True
                self.dones = True
            if "bat1" in self.sides_hit:
                self.dones = True

        self.total_steps += 1
        if self.total_steps >= self.max_steps:
            self.dones = True

        obs = self._get_obs()
        done = self.dones
        info = {}
        return obs, self.success, done, info

    def _get_obs(self):
        step_obs = {}
        for obj_name, obj_id in zip(self.object_names, self.object_ids):
            step_obs[obj_name] = {
                "xpos": self.mj_data.xpos[obj_id].copy(),
                "xquat": self.mj_data.xquat[obj_id].copy(),
            }
        for cam_name, cam_id in zip(self.cam_names, self.cam_ids):
            self.cam_renderer.update_scene(self.mj_data, camera=cam_id)
            image = self.cam_renderer.render()
            img = np.array(image).astype(np.uint8)
            step_obs[cam_name] = img

        return step_obs

    def render(self, mode='human'):
        if mode == 'rgb_array':
            raise NotImplementedError
            # return self._render_camera('left_cam')
        elif mode == 'human':
            if not hasattr(self, 'viewer'):
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            self.viewer.sync()
        else:
            raise NotImplementedError

    def close(self):
        # self.renderer.close()
        self.viewer.close()

