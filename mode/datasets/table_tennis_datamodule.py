import logging
import os
from pathlib import Path
from typing import Dict, List, Union, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import Dataset, DataLoader
import torchvision

import mode
from mode.datasets.utils.episode_utils import load_dataset_statistics
from mode.datasets.utils.shared_memory_utils import load_shm_lookup, save_shm_lookup, SharedMemoryLoader

logger = logging.getLogger(__name__)


# class TTDataset(Dataset):
#
#     def __init__(self,
#                  root_data_dir:str,
#                  action_seq_len:int,
#                  obs_seq_len: int = 1,
#                  transforms=None,
#                  task_description=None,
#                  pad:str = "repeat",
#                  **kwargs
#                  ):
#
#         self.root_dir = Path(root_data_dir)
#         self.action_seq_len = action_seq_len
#         self.obs_seq_len = obs_seq_len
#         self.transforms = hydra.utils.instantiate(transforms)
#         self.pad = pad
#
#         self.states_actions_dir = os.path.join(self.root_dir, "recordings")
#         self.images_dir = os.path.join(self.root_dir, "images")
#
#         self._build_epsode_lookup(self.root_dir)
#
#         self.task_description = task_description
#
#     def _build_epsode_lookup(self, root_dataset_dir: Path) -> np.ndarray:
#
#         assert root_dataset_dir.is_dir()
#         data_info = np.load(root_dataset_dir / "recordings/info.npz")
#         self.episode_lookup = data_info["episode_lookup"]
#         self.episode_start_end = data_info["episode_start_end"]
#         self.total_frames =  data_info["total_frames"]
#
#     def __len__(self):
#         return self.total_frames
#
#     def __getitem__(self, index: Union[int, Tuple[int, int, int]]):
#         # second option, curren index, action forward size, obs backward size
#
#         if isinstance(index, int):
#             sequence = self._get_sequence(index, self.action_seq_len)
#         else:
#             raise NotImplementedError
#         if self.pad is not None:
#             sequence = self._pad_sequence(sequence, self.pad)
#
#         sequence["idx"] = index
#         return sequence
#
#     def _get_sequence(self, index, fore_size, back_size=0):
#
#         episode, index_, ep_len = self._load_episode(index)
#
#         seq_dict = {}
#         seq_dict["rgb_obs"] = {}
#         seq_dict["rgb_obs"]["rgb_static"] = torch.from_numpy(episode["images"]["left_cam"][index_]/255.0).permute(2,0,1)
#         seq_dict["rgb_obs"]["rgb_gripper"] = torch.from_numpy(episode["images"]["right_cam"][index_]/255.0).permute(2,0,1)
#         robo_pos = torch.from_numpy(episode["state_action"]["obs"].item()["bat1"]["xpos"][index_])
#         robo_quat = torch.from_numpy(episode["state_action"]["obs"].item()["bat1"]["xquat"][index_])
#         seq_dict["robot_obs"] = torch.cat([robo_pos, robo_quat], dim=0)[None,...]
#         # seq_dict["robot_obs"] = np.concatenate([episode["state_action"]["obs"]["bat1"]["xpos"][index_],
#         #                                        episode["state_action"]["obs"]["bat1"]["xquat"][index_]])
#         # seq_dict["robot_obs"] = torch.from_numpy(seq_dict["robot_obs"])
#
#         index_right = index_ + fore_size if index_ + fore_size -1 <ep_len else ep_len
#         acts_pos = torch.tensor(episode["state_action"]["input_data"].item()["pos"][index_:index_right])
#         acts_euler = torch.from_numpy(episode["state_action"]["input_data"].item()["rot"][index_:index_right])
#         seq_dict["actions"] = torch.cat([acts_pos, acts_euler], dim=1)
#         # seq_dict["actions"] = np.concatenate([episode["input_data"]["pos"][index_:index_right],
#         #                                       episode["input_data"]["rot"][index_:index_right]],
#         #                                       axis=1)
#         # seq_dict["actions"] = torch.from_numpy(seq_dict["actions"])
#
#         seq_dict = self.apply_transform(seq_dict)
#
#         seq_dict["lang_text"] = self.task_description
#
#         return seq_dict
#
#     def _load_episode(self, index):
#         # according to global index to find the needed epsiode, load images and state_actions
#         episode = {}
#         ep_idx = self.episode_lookup[index]
#         episode["state_action"] = np.load(os.path.join(self.states_actions_dir, f"record_{ep_idx}.npz"), allow_pickle=True)
#         episode["images"] = np.load(os.path.join(self.images_dir, f"record_{ep_idx}.npz"), allow_pickle=True)
#         index_ = index - self.episode_start_end[ep_idx][0]
#         ep_len = self.episode_start_end[ep_idx][1] - self.episode_start_end[ep_idx][0]+1
#         return episode, index_, ep_len
#
#     def _pad_sequence(self, sequence, padding_mode):
#         # for simple use
#
#         pad_size = self.action_seq_len - len(sequence["actions"])
#         if pad_size <= 0:
#             return sequence
#         if padding_mode == "repetion":
#             sequence["actions"] = self._pad_with_repetition(sequence["actions"], pad_size)
#         else:
#             sequence["actions"] = self._pad_with_zeros(sequence["actions"], pad_size)
#
#         return sequence
#
#     @staticmethod
#     def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
#         """
#         Pad a sequence Tensor by repeating last element pad_size times.
#
#         Args:
#             input_tensor: Sequence to pad.
#             pad_size: Number of frames to pad.
#
#         Returns:
#             Padded Tensor.
#         """
#         last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0)
#         padded = torch.vstack((input_tensor, last_repeated))
#         return padded
#
#     @staticmethod
#     def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
#         """
#         Pad a Tensor with zeros.
#
#         Args:
#             input_tensor: Sequence to pad.
#             pad_size: Number of frames to pad.
#
#         Returns:
#             Padded Tensor.
#         """
#         zeros_repeated = torch.repeat_interleave(
#             torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0), repeats=pad_size, dim=0
#         )
#         padded = torch.vstack((input_tensor, zeros_repeated))
#         return padded
#
#     def apply_transform(self, data, trian=True):
#
#         if trian:
#             transforms = self.transforms["train"]
#         else:
#             transforms = self.transforms["test"]  # ?
#
#         for key in data["rgb_obs"]:
#             x = data["rgb_obs"][key][None, ...]  # ?
#             for transform in transforms[key]:
#                 x = transform(x)
#             data["rgb_obs"][key] = x
#
#         return data
#
#
#
# class TTDataModule(pl.LightningDataModule):
#
#     def __init__(self,
#                  datasets: DictConfig,
#                  transforms: DictConfig,
#                  num_workers: int = 12,
#                  shuffle_val: bool = False,
#                  task_embedding_format: str = "clip",
#                  split_ratio: float = 0.0,
#                  **kwargs):
#
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
#
#
#         dataset = hydra.utils.instantiate(datasets_cfg)
#
#         val_datasets = {"lang": dataset}
#
#         return val_datasets, val_datasets
#
#     def train_dataloader(self):
#         return {
#             key: DataLoader(
#                 dataset,
#                 batch_size=self.datasets_cfg.batch_size,
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
#                 batch_size=self.datasets_cfg.batch_size,
#                 num_workers=self.num_workers,
#                 shuffle=False,
#                 pin_memory=True,
#             ) for key, dataset in self.val_datasets.items()
#         }
#
#     # def split_trajectories(num_trajectories, train_ratio=0.8):
#     #     # where is it used?
#     #
#     #     num_train = int(num_trajectories * train_ratio)
#     #     num_val = num_trajectories - num_train
#     #     trajectory_indices = list(range(num_trajectories))
#     #     random.shuffle(trajectory_indices)
#     #
#     #     train_indices = trajectory_indices[:num_train]
#     #     val_indices = trajectory_indices[num_train:]
#     #
#     #     return train_indices, val_indices




class TTDataset(Dataset):

    def __init__(self,
                 root_data_dir:str,
                 action_seq_len:int,
                 obs_seq_len: int = 1,
                 transforms=None,
                 task_description=None,
                 pad:str = "repeat",
                 **kwargs
                 ):

        self.root_dir = Path(root_data_dir)
        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len
        self.transforms = hydra.utils.instantiate(transforms)
        self.pad = pad

        self.states_actions_dir = os.path.join(self.root_dir, "recordings")
        self.images_dir = os.path.join(self.root_dir, "images")

        self._build_epsode_lookup(self.root_dir)

        self.task_description = task_description

        # pre_load
        self.states_actions_data = dict(np.load(self.root_dir / "recording.npz", allow_pickle=True))

        # cache
        self._cached_episode = None
        self._cached_ep_idx = None

    def _build_epsode_lookup(self, root_dataset_dir: Path) -> np.ndarray:

        assert root_dataset_dir.is_dir()
        data_info = np.load(root_dataset_dir / "info.npz")
        self.episode_lookup = data_info["episode_lookup"]
        self.episode_start_end = data_info["episode_start_end"]
        self.total_frames =  data_info["total_frames"]

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index: Union[int, Tuple[int, int, int]]):
        # second option, curren index, action forward size, obs backward size

        if isinstance(index, int):
            sequence = self._get_sequence(index, self.action_seq_len)
        else:
            raise NotImplementedError
        if self.pad is not None:
            sequence = self._pad_sequence(sequence, self.pad)

        sequence["idx"] = index
        return sequence

    def _get_sequence(self, index, fore_size, back_size=0):

        index_, right_bound = self._load_episode(index)

        seq_dict = {}
        seq_dict["rgb_obs"] = {}
        seq_dict["rgb_obs"]["rgb_static"] = torch.from_numpy(self._cached_episode["left_cam"][index_]).permute(2,0,1)
        seq_dict["rgb_obs"]["rgb_gripper"] = torch.from_numpy(self._cached_episode["right_cam"][index_]).permute(2,0,1)

        robo_pos = torch.from_numpy(self.states_actions_data["obs"].item()["bat1"]["xpos"][index])
        robo_quat = torch.from_numpy(self.states_actions_data["obs"].item()["bat1"]["xquat"][index])
        seq_dict["robot_obs"] = torch.cat([robo_pos, robo_quat], dim=0)[None,...]

        index_right = index + fore_size if index + fore_size-1 <=right_bound else right_bound+1
        acts_pos = torch.from_numpy(np.array(self.states_actions_data["input_data"].item()["pos"][index:index_right]))
        acts_euler = torch.from_numpy(np.array(self.states_actions_data["input_data"].item()["rot"][index:index_right]))
        seq_dict["actions"] = torch.cat([acts_pos, acts_euler], dim=1)


        seq_dict = self.apply_transform(seq_dict)

        seq_dict["lang_text"] = self.task_description

        return seq_dict

    def _load_episode(self, index):
        # according to global index to find the needed epsiode, load images and state_actions
        # episode = {}
        ep_idx = self.episode_lookup[index]

        # episode["images"] = np.load(os.path.join(self.images_dir, f"record_{ep_idx}.npz"), allow_pickle=True)
        if self._cached_ep_idx != ep_idx:
            self._cached_episode = np.load(os.path.join(self.images_dir, f"record_{ep_idx}.npz"))
            self._cached_ep_idx = ep_idx

        index_ = index - self.episode_start_end[ep_idx][0]
        right_bound = self.episode_start_end[ep_idx][1]
        return index_, right_bound

    def _pad_sequence(self, sequence, padding_mode):
        # for simple use

        pad_size = self.action_seq_len - len(sequence["actions"])
        if pad_size <= 0:
            return sequence
        if padding_mode == "repetion":
            sequence["actions"] = self._pad_with_repetition(sequence["actions"], pad_size)
        else:
            sequence["actions"] = self._pad_with_zeros(sequence["actions"], pad_size)

        return sequence

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0)
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def apply_transform(self, data, trian=True):

        if trian:
            transforms = self.transforms["train"]
        else:
            transforms = self.transforms["test"]  # ?

        for key in data["rgb_obs"]:
            x = data["rgb_obs"][key][None, ...]  # ?
            for transform in transforms[key]:
                x = transform(x)
            data["rgb_obs"][key] = x

        return data



class TTDataModule(pl.LightningDataModule):

    def __init__(self,
                 datasets: DictConfig,
                 transforms: DictConfig,
                 num_workers: int = 12,
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


        dataset = hydra.utils.instantiate(datasets_cfg)

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

    # def split_trajectories(num_trajectories, train_ratio=0.8):
    #     # where is it used?
    #
    #     num_train = int(num_trajectories * train_ratio)
    #     num_val = num_trajectories - num_train
    #     trajectory_indices = list(range(num_trajectories))
    #     random.shuffle(trajectory_indices)
    #
    #     train_indices = trajectory_indices[:num_train]
    #     val_indices = trajectory_indices[num_train:]
    #
    #     return train_indices, val_indices