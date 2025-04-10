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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow_datasets as tfds
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

import mode
from mode.datasets.utils.episode_utils import load_dataset_statistics
from mode.datasets.utils.shared_memory_utils import load_shm_lookup, save_shm_lookup, SharedMemoryLoader

import time

logger = logging.getLogger(__name__)


def load_npz_dict(path):
    data = np.load(path, allow_pickle=True)
    return {k: v.item() if isinstance(v, np.ndarray) and v.dtype == object else v
            for k, v in data.items()}


class TTDataset(Dataset):

    def __init__(self,
                 metadata:str,
                 dataset_name:str="table_tennis",
                 action_seq_len:int = 120,
                 obs_seq_len: int = 1,
                 transforms=None,
                 task_description=None,
                 pad:str = "repeat",
                 **kwargs
                 ):

        ds = tfds.load(dataset_name, split="train", shuffle_files=False)
        self._dataset = sorted(tfds.as_numpy(ds), key=lambda ex:ex["episode_metadata"]["episode_id"])

        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len
        self.transforms = hydra.utils.instantiate(transforms)
        self.pad = pad

        self.task_description = task_description

        meta_info = load_npz_dict(metadata)
        self.episode_lookup = meta_info["episode_lookup"]
        self.episode_start_end = meta_info["episode_start_end"]
        self.total_frames = meta_info["total_frames"]

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

        index_, right_bound, epi = self._get_episode_index(index)

        episode = list(self._dataset[epi]["steps"])

        seq_dict = {}
        seq_dict["rgb_obs"] = {}
        seq_dict["rgb_obs"]["rgb_static"] = torch.from_numpy(episode[index_]["observation"]["left_cam"]).permute(2,0,1)
        seq_dict["rgb_obs"]["rgb_gripper"] = torch.from_numpy(episode[index_]["observation"]["right_cam"]).permute(2,0,1)

        seq_dict["robot_obs"] = torch.from_numpy(episode[index_]["observation"]["state"])[None, ...]

        index_right = index_ + fore_size -1 if index_ + fore_size-1 <=right_bound else right_bound
        actions = np.array([episode[i]["action"] for i in range(index_, index_right+1)])
        seq_dict["actions"] = torch.from_numpy(actions)

        seq_dict = self.apply_transform(seq_dict)

        seq_dict["lang_text"] = self.task_description

        return seq_dict

    def _get_episode_index(self, index):
        ep_idx = self.episode_lookup[index]
        start, end = self.episode_start_end[ep_idx]
        return index - start, end-start, ep_idx

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
