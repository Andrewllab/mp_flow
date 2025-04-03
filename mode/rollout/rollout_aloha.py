from collections import defaultdict
from functools import partial, reduce
import logging
from operator import add
from typing import Any, Dict, List, Tuple
import time
import gc

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, Trainer
import torch
import torch.distributed as dist
import einops
from tqdm import tqdm
import gymnasium as gym
import gym_aloha

from mode.datasets.base_dataset import get_validation_window_size
from mode.rollout.rollout_video import RolloutVideo
from mode.utils.utils import get_portion_of_batch_ids

log_print = logging.getLogger(__name__)


def process_obs(env_obs, transforms, lang_embed=None, lang_text=None, device="cuda"):
    return_obs = translate_obs(env_obs, device)
    return_obs = apply_obs_transforms(return_obs, transforms)

    goal = {}
    goal["lang_text"] = lang_text

    return return_obs, goal


def translate_obs(env_obs, device):
    state = torch.from_numpy(env_obs["agent_pos"]).to(device)
    state = state.to(torch.float32)
    image = torch.from_numpy(env_obs["pixels"]["top"]).to(device)
    image = einops.rearrange(image, 'h w c -> c h w')
    image = image.to(torch.float32) / 255


    state = state.unsqueeze(0)
    # image = image.unsqueeze(0)
    image = image[None, ...]

    obs = dict()
    obs["rgb_obs"] = dict()
    obs["rgb_obs"]["rgb_static"] = image
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


def process_action(action_):
    # policy predicts [12 joints, 2 gripper] -> env_action [6 joints, 1 gripper, 6 joints, 1 gripper]

    if isinstance(action_, torch.Tensor):
        action_ = action_.cpu().numpy()

    action = np.zeros_like(action_)
    action[..., :6] = action_[..., :6]
    action[..., 6] = action_[..., -2]
    action[..., 7:13] = action_[..., 6:12]
    action[..., 13] = action_[..., 13]

    return action


def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    log_print.info(*args, **kwargs)

def gather_results(local_results):
    """
    Collect eval results from all processes and average them.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return local_results
    world_size = torch.distributed.get_world_size()
    print(f"Gathering results from {world_size} GPUs...")  # Logging the number of GPUs

    # Log local results
    print(f"Local results before gathering: {local_results}")

    results = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(results, local_results)

    # Check that all lists have the same length
    if not all(len(lst) == len(local_results) for lst in results):
        raise ValueError("All result lists must have the same length")

    # Calculate the average of each corresponding element
    averaged_results = [sum(values) / world_size for values in zip(*results)]

    # Log and return the averaged results
    print(f"Averaged results: {averaged_results}")
    return averaged_results


class RolloutAloha(Callback):

    def __init__(self,
                 env_cfg: DictConfig,
                 skip_epochs: int,
                 rollout_freq,
                 n_eval,
                 max_steps,
                 num_procs,
                 use_mp,
                 empty_cache,
                 num_videos,
                 debug,
                 lang_text,
                 ):

        self.num_procs = num_procs
        self.use_mp = use_mp
        self.n_eval = n_eval
        self.max_steps = max_steps
        self.env_cfg = env_cfg

        self.skip_epochs = skip_epochs
        self.rollout_freq = rollout_freq
        self.num_videos = num_videos

        self.device = None
        self.rank = None
        self.empty_cache = empty_cache
        self.debug = debug

        self.lang_text = lang_text


    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule,):

        self.device = pl_module.device

        transforms = trainer.datamodule.transforms["val"]
        self.transforms = hydra.utils.instantiate(transforms)

        # must log the metric used to select checkpoint from the first epoch!
        if pl_module.current_epoch == 0 and self.skip_epochs > 0:
            # for i in range(self.num_tasks):
            #    pl_module.log(f"eval_lh/sr_chain_{i}", torch.tensor(0.0), on_step=False, sync_dist=True)
            pl_module.log("eval_success_rate", torch.tensor(0.0), on_step=False, sync_dist=True)

        elif pl_module.current_epoch == self.skip_epochs or \
                ((pl_module.current_epoch - self.skip_epochs) >= 0 and (pl_module.current_epoch - self.skip_epochs) % self.rollout_freq == 0):
            successes = self.evaluate_policy(pl_module, store_video=self.num_videos)
            successes =  gather_results(successes)

            log_rank_0(f"eval_success_rate {torch.tensor(successes)}")
            pl_module.log("eval_success_rate", torch.tensor(successes), on_epoch=True, sync_dist=True)
            # trainer.callback_metrics["eval_success_rate"] = torch.tensor(successes)

            print("evaluation done")

    def evaluate_policy(self, model, store_video=False):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # if goal-conditioned IL, modify here to evaluate multiple tasks
        log_rank_0(f"starts to evaluate")
        sucessses = self.evaluate_parallel(model, rank, world_size, store_video=store_video)

        return sucessses

    def evaluate_parallel(self, model, rank, world_size, store_video=False):

        rollouts_per_gpu = self.n_eval // world_size
        extra_rollouts = self.n_eval % world_size
        start_rollout = rank * rollouts_per_gpu + min(rank, extra_rollouts)
        end_rollout = start_rollout + rollouts_per_gpu + (1 if rank < extra_rollouts else 0)
        eval_loop_num =end_rollout - start_rollout


        # env_num = min(self.num_procs, eval_loop_num) if self.use_mp else 1  # for workers > 1, parallel gym_env wrapper needed
        env_num = 1

        env_creation = False
        count = 0
        while not env_creation and count < 5:
            try:
                if env_num == 1:
                    # env = DummyVectorEnv(
                    #     [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    # )
                    # env = gym.make("gym_aloha/AlohaTransferCube-v0",
                    #                obs_type="pixels_agent_pos",
                    #                max_episode_steps=500, )

                    env_cfg = OmegaConf.to_container(self.env_cfg, resolve=True)
                    env = gym.make(env_cfg["id"], obs_type=env_cfg["obs_type"], max_episode_steps=env_cfg["max_episode_steps"])
                    # always include aloha_gym to create env, althrough only calling gymnasium interface
                # else:
                #     env = SubprocVectorEnv(
                #         [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                #     )  # not implemented
                env_creation = True
            except:
                time.sleep(5)
                count += 1
        if count >= 5:
            raise Exception("Failed to create environment")

        num_successes = 0
        for i in tqdm(range(start_rollout, end_rollout), desc="evluating"):
            # if store_video:

            # frames = []
            obs_, info = env.reset(seed=rank*100+i*10)
            step = 0
            while step < self.max_steps:
                step += 1
                obs, goal = process_obs(obs_, transforms=self.transforms, lang_text=self.lang_text)
                actions = model.step(obs, goal)
                actions = process_action(actions)
                obs_, reward, terminated, truncated, info = env.step(actions)

                if terminated:
                    num_successes += 1
                    break

        env.close()
        gc.collect()

        return [num_successes/eval_loop_num]


