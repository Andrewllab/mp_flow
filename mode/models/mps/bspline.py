from functools import wraps

import torch
from addict import Dict

from mp_pytorch.mp import MPFactory
from mp_pytorch.util import add_expand_dim
# from mode.utils.utils import timeit


def autocast_float32(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            return fn(*args, **kwargs)
    return wrapped

def normalize(tensor, min_val, max_val):
    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()

    # Normalize the tensor to [0, 1]
    assert torch.all(
        tensor >= min_val - 1e-3), "Input tensor has values below min_val"
    assert torch.all(
        tensor <= max_val + 1e-3), "Input tensor has values above max_val"
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    normalized_tensor = torch.clamp(normalized_tensor, 0, 1)

    return normalized_tensor

def unnormalize(normalized_tensor, min_val, max_val):

    tensor = normalized_tensor * (max_val - min_val) + min_val
    return tensor


class BSpline(torch.nn.Module):

    def __init__(self, num_dof, num_basis=10, degree_p=4, dtype=torch.float32,
                 device="cpu", seq_len=60, frequency=30, **kwargs):
        super().__init__()

        self.num_dof = num_dof
        self.mp_config = Dict()
        self.mp_config.mp_type = "uni_bspline"
        self.mp_config.dtype = dtype
        self.mp_config.device = device
        self.mp_config.num_dof = num_dof
        self.mp_config.tau = seq_len / frequency
        self.mp_config.mp_args.num_basis = num_basis
        self.mp_config.mp_args.degree_p = degree_p
        self.mp_config.mp_args.init_condition_order = kwargs.get("init_condition_order", 0)
        self.mp_config.mp_args.end_condition_order = kwargs.get("end_condition_order", 0)
        self.mp_config.mp_args.weights_scale = kwargs.get("weights_scale", 1)

        self.mp = MPFactory.init_mp(**self.mp_config)

        self.seq_len = seq_len
        self.frequency = frequency
        self.duration = self.mp_config.tau
        times = torch.linspace(0, self.duration, seq_len, dtype=self.dtype,
                                    device=self.device)
        self.register_buffer("times", times, persistent=False)

        # normalizer bounds making the diffuser predicted params in [-1, 1],
        # updated in every batch, should be stablized after 1 epoch training
        self.register_buffer("w_min", -2.0 * torch.ones((num_dof * num_basis)))
        self.register_buffer("w_max", 2.0 * torch.ones((num_dof * num_basis)))

    @property
    def device(self):
        return self.mp.device

    @property
    def dtype(self):
        return self.mp.dtype

    @torch.no_grad()
    @autocast_float32
    def traj_to_params(self, action_sequences, update_bounds: bool = False):

        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of trajs:
        # [*add_dim, num_times, num_dof]
        #
        # Shape of learned params
        # [*add_dim, num_dof * num_basis_g]
        add_dim = list(action_sequences.shape[:-2])
        times = add_expand_dim(self.times, list(range(len(add_dim))), add_dim)

        # for t in range(1, actions.shape[-2]):
        # actions[..., t, :] = actions[..., t, :] + actions[..., t-1, :]
        absolute2current = action_sequences.cumsum(-2)
        # dictionary
        para = self.mp.learn_mp_params_from_trajs(times, absolute2current)
        if update_bounds:
            self.update_weights_bounds_per_batch(para["params"])

        # normalize weights
        params = normalize(para["params"], self.w_min, self.w_max)
        # [*add_dim, num_dof * num_basis] -> [*add_dim, num_dof, num_basis]
        params = params.reshape(*self.mp.add_dim, self.mp.num_dof, -1)
        # -> [*add_dim, num_basis, num_dof]
        params = torch.einsum("...ji->...ij", params)

        para["params"] = params

        # debug
        # params_ = torch.einsum('...ji->...ij', para["params"])
        # add_dim_ = list(params_.shape[:-2])
        # params_ = params_.reshape(*add_dim_, -1)
        # params_ = unnormalize(params_, self.w_min, self.w_max)
        # para_ = dict()
        # para_["params"] = params_
        # self.mp.update_inputs(times, **para_)
        # absolute_pre = self.mp.get_traj_pos()
        #
        # import matplotlib.pyplot as plt
        # dim = 3
        #
        # plt.plot(absolute2current[:, dim], color="blue", label="gt")
        # plt.plot(absolute_pre[:, dim], color="red", label="traj_pred")
        # plt.legend()
        # plt.show()
        #
        # rel = torch.diff(torch.tensor(absolute2current), dim=-2,
        #                  prepend=torch.zeros([*add_dim, 1, self.mp.num_dof],
        #                                      dtype=self.dtype,
        #                                      device=self.device))
        # rel_pre = torch.diff(torch.tensor(absolute_pre), dim=-2,
        #                      prepend=torch.zeros([*add_dim, 1, self.mp.num_dof],
        #                                          dtype=self.dtype,
        #                                          device=self.device))
        # plt.plot(action_sequences[:, dim], color="blue", label="rel_gt")
        # plt.plot(rel[:, dim], color="red", label="rel_rec")
        # plt.plot(rel_pre[:, dim], color="green", label="rel_pre")
        # plt.legend()
        # plt.show()
        #
        # print(" ")

        # return para["params"]
        return para

    @torch.no_grad()
    def get_traj(self, para):

        # [*add_dim, num_basis, num_dof] -> [*add_dim, num_dof, num_basis]
        params = torch.einsum('...ji->...ij', para["params"])
        add_dim = list(params.shape[:-2])
        # -> [*add_dim, num_dof * num_basis]
        params = params.reshape(*add_dim, -1)
        # unnormalize
        params = unnormalize(params, self.w_min, self.w_max)

        para["params"] = params
        times = add_expand_dim(self.times, list(range(len(add_dim))), add_dim)
        self.mp.update_inputs(times, **para)
        # [*add_dim, num_times, num_dof]
        absolute2curr = self.mp.get_traj_pos()
        traj = torch.diff(absolute2curr, dim=-2, prepend=torch.zeros([*add_dim, 1, self.mp.num_dof], dtype=self.dtype, device=self.device))

        return traj

    def update_weights_bounds_per_batch(self, weights):
        weights = weights.reshape(-1, self.mp_config.num_dof * self.mp_config.mp_args.num_basis)
        batch_min = weights.min(dim=0)[0]
        batch_max = weights.max(dim=0)[0]
        smaller_mask = batch_min < (self.w_min - 1e-4)
        larger_mask = batch_max > (self.w_max + 1e-4)
        if torch.any(smaller_mask):
            self.w_min[smaller_mask] = batch_min[smaller_mask]
        if torch.any(larger_mask):
            self.w_max[larger_mask] = batch_max[larger_mask]

class BSplineD(BSpline):

    def __init__(self, num_dof, num_basis=10, degree_p=4, dtype=torch.float32,
                 device="cpu", seq_len=60, frequency=30, digit_dims=1, **kwargs):
        super(BSplineD, self).__init__(num_dof=num_dof-digit_dims,
                                       num_basis=num_basis, degree_p=degree_p, dtype=dtype,
                                       device=device, seq_len=seq_len, frequency=frequency, **kwargs)
        self.num_dof = num_dof
        self.digit_dims = digit_dims
        self.mpd_config = Dict()
        self.mpd_config.mp_type = "uni_bspline"
        self.mpd_config.dtype = dtype
        self.mpd_config.device = device
        self.mpd_config.num_dof = digit_dims
        self.mpd_config.tau = seq_len / frequency
        self.mpd_config.mp_args.num_basis = num_basis
        self.mpd_config.mp_args.degree_p = 0

        self.mpd = MPFactory.init_mp(**self.mpd_config)

    @torch.no_grad()
    @autocast_float32
    def traj_to_params(self, action_sequences, update_bounds: bool = False):

        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of trajs:
        # [*add_dim, num_times, num_dof]
        #
        # Shape of learned params
        # [*add_dim, num_dof * num_basis_g]

        para_ = super(BSplineD, self).traj_to_params(action_sequences[..., :-self.digit_dims], update_bounds)

        add_dim = list(action_sequences.shape[:-2])
        times = add_expand_dim(self.times, list(range(len(add_dim))), add_dim)

        para_d = self.mpd.learn_mp_params_from_trajs(times, action_sequences[..., self.mp.num_dof:])
        params_d = para_d["params"].reshape(*self.mpd.add_dim, self.mpd.num_dof, -1)
        # -> [*add_dim, num_basis, num_dof]
        params_d = torch.einsum('...ji->...ij', params_d)

        para_["params"] = torch.cat([para_["params"], params_d], dim=-1)

        return para_

    # @timeit
    @torch.no_grad()
    def get_traj(self, para):

        para_ = dict()
        para_["params"] = para["params"][..., :-self.digit_dims]
        traj_ = super(BSplineD, self).get_traj(para_)

        params_d = para["params"][..., self.mp.num_dof:]
        # -> [*add_dim, num_dof, num_basis]
        params_d = torch.einsum('...ji->...ij', params_d)
        add_dim = list(params_d.shape[:-2])
        params_d = params_d.reshape(*add_dim, -1)

        times = add_expand_dim(self.times, list(range(len(add_dim))), add_dim)
        self.mpd.update_inputs(times, **{"params": params_d})
        traj_d = self.mpd.get_traj_pos()

        traj = torch.cat([traj_, traj_d], dim=-1)

        return traj

