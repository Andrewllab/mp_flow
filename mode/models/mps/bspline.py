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

    @property
    def device(self):
        return self.mp.device

    @property
    def dtype(self):
        return self.mp.dtype

    @autocast_float32
    def traj_to_params(self, action_sequences):

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

        # dictionary
        para = self.mp.learn_mp_params_from_trajs(times, action_sequences)
        # Reshape params
        # [*add_dim, num_dof * num_basis] -> [*add_dim, num_dof, num_basis]
        params = para["params"].reshape(*self.mp.add_dim, self.mp.num_dof, -1)
        params = torch.einsum('...ji->...ij', params)
        para["params"] = params

        # return para["params"]
        return para

    def get_traj(self, para):

        params = torch.einsum('...ji->...ij', para["params"])
        add_dim = list(params.shape[:-2])
        params = params.reshape(*add_dim, -1)
        para["params"] = params
        # add_dim = list(para["params"].shape[:-1])
        times = add_expand_dim(self.times, list(range(len(add_dim))), add_dim)
        self.mp.update_inputs(times, **para)

        traj = self.mp.get_traj_pos()

        return traj


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

    @autocast_float32
    def traj_to_params(self, action_sequences):

        # Shape of times:
        # [*add_dim, num_times]
        #
        # Shape of trajs:
        # [*add_dim, num_times, num_dof]
        #
        # Shape of learned params
        # [*add_dim, num_dof * num_basis_g]

        para_ = super(BSplineD, self).traj_to_params(action_sequences[..., :-self.digit_dims])

        add_dim = list(action_sequences.shape[:-2])
        times = add_expand_dim(self.times, list(range(len(add_dim))), add_dim)

        para_d = self.mpd.learn_mp_params_from_trajs(times, action_sequences[..., self.mp.num_dof:])
        params_d = para_d["params"].reshape(*self.mpd.add_dim, self.mpd.num_dof, -1)
        # -> [*add_dim, num_basis, num_dof]
        params_d = torch.einsum('...ji->...ij', params_d)

        para_["params"] = torch.cat([para_["params"], params_d], dim=-1)

        return para_

    # @timeit
    def get_traj(self, para):

        # -> [*add_dim, num_dof, num_basis]
        params = torch.einsum('...ji->...ij', para["params"])
        add_dim = list(params.shape[:-2])
        params_ = params[..., :-self.digit_dims, :].reshape(*add_dim, -1)
        params_d = params[..., self.mp.num_dof:, :].reshape(*add_dim, -1)

        times = add_expand_dim(self.times, list(range(len(add_dim))), add_dim)
        self.mp.update_inputs(times, **{"params": params_})
        self.mpd.update_inputs(times, **{"params": params_d})

        traj_ = self.mp.get_traj_pos()
        traj_d = self.mpd.get_traj_pos()
        traj = torch.cat([traj_, traj_d], dim=-1)

        return traj

