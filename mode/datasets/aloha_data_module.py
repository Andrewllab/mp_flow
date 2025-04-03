import os

import torch
from numba.scripts.generate_lower_listing import description
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler, random_split, Dataset
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
import random
import numpy as np

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDataset


# m_demos dataset.meta.total_episodes  dataset.num_episodes
# total_num_sequences  meta.total_frames  dataset.num_frames
# description text: dataset.meta.tasks[0]
# camera_key = dataset.meta.camera_keys[0]
# delta_timesteps = {
#    dataset.meta.camera_keys[0]: [0],
#    "observation.state": [0],
#    "action" : [t/ dataset.fps for t in range(20)]
#   }
#

# class AlohaDataset(Dataset):
#
#     def __init__(self,
#                  lebot_dataset,
#                  task_description,
#                  obs_sq_len: int=1,
#                  act_sq_len: int=1,
#                  transforms=None):
#
#         self.obs_sq_len = obs_sq_len
#         self.act_sq_len = act_sq_len
#         self.transforms = hydra.utils.instantiate(transforms)
#         self.lebot_dataset = lebot_dataset
#
#         # self.lebot_dataset.goal_mode = "last"  # what is it about?
#         # self.task_emb = None
#         self.task_description = task_description
#         self.n_demos = self.lebot_dataset.num_episodes
#         self.total_num_sequences = self.lebot_dataset.num_frames
#
#     def __len__(self):
#         return len(self.lebot_dataset)
#
#     def __getitem__(self, idx):
#         data_dict = {}
#         return_dict = self.lebot_dataset[idx]
#         data_dict["lang"] = self.select_translate_data(return_dict)
#
#         if self.transforms:
#             data_dict = self.apply_transform(data_dict["lang"])
#         data_dict["idx"] = idx
#
#         return data_dict
#
#     def apply_transform(self, data, trian=True):
#
#         if trian:
#             transforms = self.transforms["train"]
#         else:
#             transforms = self.transforms["test"] #?
#
#         for key in data["rgb_obs"]:
#             x = data["rgb_obs"][key][None,...]  # ?
#             for transform in transforms[key]:
#                 x = transform(x)
#             data["rgb_obs"][key] = x
#
#         return data
#
#     def select_translate_data(self, data):
#
#         return_dict = {}
#         # for key in self.lebot_datset.meta.camera_keys:
#             # return_dict[key] = data[key]
#         return_dict["rgb_obs"] = {}
#         return_dict["rgb_obs"]["rgb_static"] = data["observation.images.top"]
#         return_dict["robot_obs"] = data["observation.state"]
#         action_data = data["action"]
#         # move the gripper dimension to the end
#         original_ind = list(range(14))
#         new_order = [i for i in original_ind if i not in [6, 13]] + [6, 13]
#         action_data = action_data[..., new_order]
#         return_dict["actions"] = action_data
#
#         return_dict["lang_text"] = data["task"]
#         return_dict["depth_obs"] = {}
#
#         return return_dict
#
#
# class AlohaDataModule(pl.LightningDataModule):
#
#     def __init__(self,
#                  datasets: DictConfig,
#                  num_workers: int = 12,
#                  transforms: DictConfig = None,
#                  shuffle_val: bool = False,
#                  task_embedding_format: str = "clip",
#                  split_ratio: float = 0.0,
#                  **kwargs):
#         super().__init__()
#         self.datasets_cfg = datasets
#         self.num_workers = num_workers
#         self.transforms = transforms
#         self.shuffle_val = shuffle_val
#         self.train_datasets = []
#         self.val_datasets = []
#         self.train_sampler = None
#         self.val_sampler = None
#         self.modalities = []
#         self.task_embedding_format = task_embedding_format
#         self.split_ratio = split_ratio
#
#     def setup(self, stage):
#
#         self.train_datasets, self.val_datasets = self._initialize_datasets(self.datasets_cfg)
#         self.modalities.append('lang')
#
#     def _initialize_datasets(self, datasets_cfg):
#         # train_datasets = []
#         # descriptions = []
#
#         fps = 50
#         delta_timestamps = {
#             "observation.images.top" : [0],
#             "observation.state": [0],
#             "action": [t / fps for t in range(self.datasets_cfg.act_seq_len)],
#         }
#         lebot_dataset = LeRobotDataset(repo_id=self.datasets_cfg.repo_id,
#                                  delta_timestamps=delta_timestamps)
#
#         dataset = AlohaDataset(lebot_dataset=lebot_dataset,
#                                task_description=lebot_dataset.meta.tasks[0],
#                                act_sq_len=self.datasets_cfg.act_seq_len,
#                                transforms=self.datasets_cfg.transforms
#                                )
#
#         val_datasets = {"lang": dataset}
#
#         return val_datasets, val_datasets
#
#     def train_dataloader(self):
#         return {
#             key: DataLoader(
#                 dataset,
#                 batch_size=self.datasets_cfg.lang_dataset.batch_size,
#                 num_workers=self.num_workers,
#                 pin_memory=True,
#                 shuffle=True,
#             )
#             for key, dataset in self.train_datasets.items()
#         }
#
#     def val_dataloader(self):
#         return {
#             key: DataLoader(
#                 dataset,
#                 batch_size=self.datasets_cfg.lang_dataset.batch_size,
#                 num_workers=self.num_workers,
#                 shuffle=False,
#                 pin_memory=True,
#             ) for key, dataset in self.val_datasets.items()
#         }
#
#     def split_trajectories(num_trajectories, train_ratio=0.8):
#         # where is it used?
#
#         num_train = int(num_trajectories * train_ratio)
#         num_val = num_trajectories - num_train
#         trajectory_indices = list(range(num_trajectories))
#         random.shuffle(trajectory_indices)
#
#         train_indices = trajectory_indices[:num_train]
#         val_indices = trajectory_indices[num_train:]
#
#         return train_indices, val_indices



class AlohaDataset(Dataset):

    def __init__(self,
                 lebot_dataset_cfg,
                 transforms=None,
                 batch_size=None,
                 num_workers=None,
                 **kwargs):


        self.lebot_dataset_cfg = lebot_dataset_cfg
        act_seq_len = self.lebot_dataset_cfg.act_seq_len
        obs_sq_len = self.lebot_dataset_cfg.obs_seq_len
        fps = self.lebot_dataset_cfg.fps
        delta_timestamps = {
            "observation.images.top": [t/fps for t in range(-obs_sq_len+1, 1)],
            "observation.state": [t/fps for t in range(-obs_sq_len+1, 1)],
            "action": [t/fps for t in range(act_seq_len)],
        }
        self.lebot_dataset = LeRobotDataset(repo_id=self.lebot_dataset_cfg.repo_id,
                                       delta_timestamps=delta_timestamps)

        self.transforms = hydra.utils.instantiate(transforms)

        self.task_description = kwargs["task_description"]
        self.n_demos = self.lebot_dataset.num_episodes
        self.total_num_sequences = self.lebot_dataset.num_frames

    def __len__(self):
        return len(self.lebot_dataset)

    def __getitem__(self, idx):
        data_dict = {}
        return_dict = self.lebot_dataset[idx]
        data_dict["lang"] = self.select_translate_data(return_dict)

        if self.transforms:
            data_dict = self.apply_transform(data_dict["lang"])
        data_dict["idx"] = idx

        return data_dict

    def apply_transform(self, data, trian=True):

        if trian:
            transforms = self.transforms["train"]
        else:
            transforms = self.transforms["test"] #?

        for key in data["rgb_obs"]:
            x = data["rgb_obs"][key][None,...]  # ?
            for transform in transforms[key]:
                x = transform(x)
            data["rgb_obs"][key] = x

        return data

    def select_translate_data(self, data):

        return_dict = {}
        # for key in self.lebot_datset.meta.camera_keys:
            # return_dict[key] = data[key]
        return_dict["rgb_obs"] = {}
        return_dict["rgb_obs"]["rgb_static"] = data["observation.images.top"]
        return_dict["robot_obs"] = data["observation.state"]
        action_data = data["action"]
        # move the gripper dimension to the end
        original_ind = list(range(14))
        new_order = [i for i in original_ind if i not in [6, 13]] + [6, 13]
        action_data = action_data[..., new_order]
        return_dict["actions"] = action_data

        return_dict["lang_text"] = data["task"]
        return_dict["depth_obs"] = {}

        return return_dict


class AlohaDataModule(pl.LightningDataModule):

    def __init__(self,
                 datasets: DictConfig,
                 num_workers: int = 12,
                 transforms: DictConfig = None,
                 shuffle_val: bool = False,
                 task_embedding_format: str = "clip",
                 split_ratio: float = 0.0,
                 **kwargs):
        super().__init__()
        self.datasets_cfg = datasets
        self.num_workers = num_workers
        self.transforms = transforms
        self.shuffle_val = shuffle_val
        self.train_datasets = []
        self.val_datasets = []
        self.train_sampler = None
        self.val_sampler = None
        self.modalities = []
        self.task_embedding_format = task_embedding_format
        self.split_ratio = split_ratio

    def setup(self, stage):

        self.train_datasets, self.val_datasets = self._initialize_datasets(self.datasets_cfg)
        self.modalities.append('lang')

    def _initialize_datasets(self, datasets_cfg):
        # train_datasets = []
        # descriptions = []

        dataset = hydra.utils.instantiate(datasets_cfg)

        # dataset = AlohaDataset(lebot_dataset=lebot_dataset,
        #                        task_description=lebot_dataset.meta.tasks[0],
        #                        act_sq_len=self.datasets_cfg.act_seq_len,
        #                        transforms=self.datasets_cfg.transforms
        #                        )

        val_datasets = {"lang": dataset}

        return val_datasets, val_datasets

    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=self.datasets_cfg.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=self.datasets_cfg.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
            ) for key, dataset in self.val_datasets.items()
        }

    def split_trajectories(num_trajectories, train_ratio=0.8):
        # where is it used?

        num_train = int(num_trajectories * train_ratio)
        num_val = num_trajectories - num_train
        trajectory_indices = list(range(num_trajectories))
        random.shuffle(trajectory_indices)

        train_indices = trajectory_indices[:num_train]
        val_indices = trajectory_indices[num_train:]

        return train_indices, val_indices