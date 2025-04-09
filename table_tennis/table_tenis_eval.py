import sys
sys.path.append("/home/i53/student/wliao/projects/mp_flow")




from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time

from omegaconf import OmegaConf

# from test_env import lang_embeddings

# sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import hydra
import numpy as np
import torch
import wandb
import torch.distributed as dist
import torch.serialization



# from mode.evaluation.multistep_sequences import get_sequences
# from mode.evaluation.utils import get_default_mode_and_env, get_env_state_for_initial_condition, join_vis_lang
from mode.evaluation.utils import load_pl_module_from_checkpoint
# from mode.rollout.rollout_video import RolloutVideo
# from mode.rollout.rollout_aloha import process_obs, process_action, apply_obs_transforms

import einops
import matplotlib.pyplot as plt

# import mode.utils.pose_process as pp

from table_tennis.table_tennis_env import TableTennisEnv
import gym_aloha
import gymnasium as gym
import imageio



# Patch `pl_load` to always load with weights_only=False
original_torch_load = torch.load

def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False  # force full unpickling
    return original_torch_load(*args, **kwargs)

torch.load = patched_load


model = load_pl_module_from_checkpoint("/home/i53/student/wliao/Desktop/tabletennis/0409_ckpt/last.ckpt")

# plot the trajectory

# dm_cfg = OmegaConf.load("/home/i53/student/wliao/projects/mp_flow/conf/datamodule/aloha_debug.yaml")
# dm = hydra.utils.instantiate(dm_cfg)
#
# dm.setup(stage="fit")
# dataloader = dm.val_dataloader()
# for batch in dataloader["lang"]:
#     break
#
#
#
# bt = 30
#
# obs = {}
# obs["rgb_obs"] = {}
# obs["rgb_obs"]["rgb_static"] = batch["rgb_obs"]["rgb_static"].to(model.device)[bt, ...][None,...]
# obs["robot_obs"] = batch["robot_obs"].to(model.device)[bt, ...][None,...]
# goal = {"lang_text": batch["lang_text"][0]}
#
# # action = model.step(obs, goal)
# # action = process_action(action)
# action = model.forward(obs, goal)
#
# traj_gt = batch["actions"][bt, :100, :].detach().cpu().numpy()
# traj_rec = action.detach().cpu().numpy()[0]
#
#
# fig, axs = plt.subplots(14, 1, figsize=(16, 20), sharex=True)
#
# for dim in range(14):
#     axs[dim].plot(traj_gt[:, dim], label="gt")
#     # axs[dim].plot(traj_mprec[ :, dim].cpu().numpy(), label="mp_reg")
#     axs[dim].plot(traj_rec[:, dim], label="rec")
#     # axs[dim].plot(traj_pre[b, :, dim], label="pre")
#     # axs[dim].plot(trajs_pred[b, :, dim].cpu().numpy(), label="pred")
#     axs[dim].set_title(f"Dimension {dim}")
#     axs[dim].legend()
#
# plt.tight_layout()
# plt.show()


def process_obs(env_obs, transforms, lang_text=None, device="cuda"):
    return_obs = translate_obs(env_obs, device)
    return_obs = apply_obs_transforms(return_obs, transforms)

    goal = {}
    goal["lang_text"] = lang_text

    return return_obs, goal


def translate_obs(env_obs, device):
    bat1_pos = torch.from_numpy(env_obs["bat1"]["xpos"]).to(device)
    bat1_quat = torch.from_numpy(env_obs["bat1"]["xquat"]).to(device)
    state = torch.cat([bat1_pos, bat1_quat])

    left_image = torch.from_numpy(env_obs["left_cam"]).to(device)
    left_image = einops.rearrange(left_image, 'h w c -> c h w')
    right_image = torch.from_numpy(env_obs["right_cam"]).to(device)
    right_image = einops.rearrange(right_image, 'h w c -> c h w')


    state = state.unsqueeze(0)
    # image = image.unsqueeze(0)
    left_image = left_image[None, ...]
    right_image = right_image[None, ...]

    obs = dict()
    obs["rgb_obs"] = dict()
    obs["rgb_obs"]["rgb_static"] = left_image
    obs["rgb_obs"]["rgb_gripper"] = right_image
    obs['robot_obs'] = state

    return obs

def apply_obs_transforms(obs, transforms):
    for key in obs['rgb_obs']:
        data = obs['rgb_obs'][key]
        if len(data.shape) == 3:
            data = data[None, ...]
        for transform in transforms[key]:
            data = transform(data)
        if len(data.shape) == 4:
            data = data[None, ...]
        obs['rgb_obs'][key] = data

    return obs










# simulate
output_dir = Path("/home/temp_store/weiran/tabletennis_out/video/modemp")
output_dir.mkdir(parents=True, exist_ok=True)

# env = gym.make("gym_aloha/AlohaTransferCube-v0",
#                obs_type="pixels_agent_pos",
#                max_episode_steps=1200,)
env = TableTennisEnv(model_path="/home/i53/student/wliao/Desktop/tabletennis/assets/table_tennis_env.xml",max_step=1000)

transforms_cfg = OmegaConf.load("/home/i53/student/wliao/projects/mp_flow/conf/datamodule/transforms/tabletennis_transforms.yaml")
transforms_cfg = transforms_cfg.val
transforms = hydra.utils.instantiate(transforms_cfg)


i = 1
# rewards = []
# frames = []

# frames.append(env.render())
successt = 0
n_eval = 50
for i in range(n_eval):

#frames = []
    observation, info = env.reset()
# frames.append(env.render())
#env.render()

    done = False
    while not done:


        goal = "hit the coming ball to the opposite table"
        obs, goal = process_obs(observation, transforms=transforms, lang_text=goal)

        action = model.step(obs, goal)
        action = action.to("cpu").numpy()

        # action_ = env.action_space.sample()
        observation, success, done, info = env.step(action)

        # frames.append(env.render())

        if success:
            print("success")
            successt += 1
        elif done:
            print("failure")


print(success/n_eval)
env.close()
# print(successt)



# # Get the speed of environment (i.e. its number of frames per second).
# fps = env.metadata["render_fps"]
# # Encode all frames into a mp4 video.
# video_path = output_dir / "rollout.mp4"
# imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
#
# print(f"Video of the evaluation is available in '{video_path}'.")




