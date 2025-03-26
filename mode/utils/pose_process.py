
import numpy as np
import torch

import robosuite.utils.transform_utils as T


def rel2abs(rel_action, ee_state):
    """

    rel_action: delta_action, the first 3-dimension is delta position, the second 3-dimension is delta orientation in
                xyz euler angle
    ee_state: current end-effector pose, the first 3-dimension is postion, the next 3-dimension is orientation in
              axis-angle notation.

    return:
        in orientation in axis-angle notation

    """
    ref = ee_state + rel_action

    for b in range(ref.shape[0]):
        for t in range(ref.shape[1]):
            ori_d = combine_aa_euler(ee_state[b, t, 3:6], rel_action[b, t, 3:6])
            ref[b, t, 3:6] = ori_d

    return ref


def combine_aa_euler(aa_ori, euler_rot):

    quat_ori = T.axisangle2quat(aa_ori)

    mat_rot = T.euler2mat(euler_rot)
    quat_rot = T.mat2quat(mat_rot)

    quat_ori_new = T.quat_multiply(quat_rot, quat_ori)

    aa_ori_new = T.quat2axisangle(quat_ori_new)

    return aa_ori_new


def get_delta_rot(desired_ori_aa, current_ori_quat):

    # precision needs to be improved

    desired_ori_quat = T.axisangle2quat(desired_ori_aa)
    desired_ori_mat = T.quat2mat(desired_ori_quat)

    current_ori_mat = T.quat2mat(current_ori_quat)

    delta_mat = np.dot(desired_ori_mat, np.linalg.inv(current_ori_mat))

    delta_euler = T.mat2euler(delta_mat)  # axes?

    return delta_euler









