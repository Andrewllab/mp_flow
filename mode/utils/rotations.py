import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle to quaternion."""
    angle = np.linalg.norm(axis_angle)
    if angle == 0:
        return np.array([1, 0, 0, 0])  # Identity quaternion
    # axis = axis_angle / angle
    # rot = R.from_rotvec(axis * angle)
    rot = R.from_rotvec(axis_angle)
    return np.array(rot.as_quat())  # Returns [x, y, z, w]


def quaternion_to_axis_angle(quat):
    """Convert quaternion to axis-angle."""

    ch = isinstance(quat, torch.Tensor)
    if ch:
        device = quat.device
        dtype = quat.dtype
        quat = quat.to("cpu").numpy()

    rot = R.from_quat(quat)
    axis_angle = rot.as_rotvec()
    axis_angle = np.array(axis_angle)

    if ch:
        axis_angle = torch.tensor(axis_angle, device=device, dtype=dtype)
    return axis_angle

def quaternion_to_euler(quat, degrees=False, order='xyz'):
    """
    Convert a quaternion to Euler angles.

    Parameters:
        quat (list or tuple): Quaternion in the form [x, y, z, w]
        degrees (bool): Whether to return angles in degrees (default: True)
        order (str): Axis rotation order, e.g., 'xyz', 'zyx' (default: 'xyz')

    Returns:
        tuple: (roll, pitch, yaw) in the specified order
    """
    r = R.from_quat(quat)
    euler = r.as_euler(order, degrees=degrees)
    return np.array(euler)

def combine_axis_angle_rotations(aa1, aa2):
    """Combine two axis-angle rotations."""

    ch = isinstance(aa2, torch.Tensor)
    if ch:
        device = aa2.device
        dtype = aa2.dtype
        aa1 = aa1.to("cpu").numpy()
        aa2 = aa2.to("cpu").numpy()

    # q1 = axis_angle_to_quaternion(aa1)
    # q2 = axis_angle_to_quaternion(aa2)

    # Quaternion multiplication: q_new = q2 * q1
    # because q2 is a delta on top of q1
    # r1 = R.from_quat(q1)
    # r2 = R.from_quat(q2)
    r1 = R.from_rotvec(aa1)
    r2 = R.from_rotvec(aa2)
    # r2 is rotation expressed in global world coordinate
    r_new = r2 * r1
    # r2 is rotation expressed in local frame attached to eef
    # r_new = r1 * r2

    r_new = np.array(r_new.as_rotvec())

    if ch:
        r_new = torch.tensor(r_new, device=device, dtype=dtype)

    return r_new

def combine_euler_rotations(euler1, euler2, order='xyz', degrees=False, body_frame=False):
    """Combine two Euler angle rotations."""
    r1 = R.from_euler(order, euler1, degrees=degrees)
    r2 = R.from_euler(order, euler2, degrees=degrees)

    if body_frame:
        # Apply second rotation in local (body) frame
        r_new = r1 * r2
    else:
        # Apply second rotation in world (global) frame
        r_new = r2 * r1

    return r_new.as_euler(order, degrees=degrees)

def delta_axis_angle(from_aa, to_aa):

    r_from = R.from_rotvec(from_aa)
    r_to = R.from_rotvec(to_aa)

    r_delta = r_to * r_from.inv()

    return r_delta.as_rotvec()


if __name__ == '__main__':
    # Example usage
    initial_axis_angle = np.array([0, 0, np.pi / 2])
    delta_axis_angle = np.array([0, 0, np.pi / 2])

    new_orientation = combine_axis_angle_rotations(initial_axis_angle, delta_axis_angle)
    print("New orientation (axis-angle):", new_orientation)

    # Convert a quaternion to axis-angle
    quat = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # 90° around Z
    aa = quaternion_to_axis_angle(quat)
    print("Axis-angle from quaternion:", aa)