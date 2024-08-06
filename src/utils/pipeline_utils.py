import math
import numpy as np

def get_orientation(vector):
    """
    Get the orientation angle from a quaternion.

    Args:
        vector (fsds.Quaternionr): Quaternion vector.

    Returns:
        float: Orientation angle in degrees.
    """
    w_val = vector.w_val
    z_val = vector.z_val
    a_acos = math.acos(w_val)
    if z_val < 0:
        angle = math.degrees(-a_acos)
    else:
        angle = math.degrees(a_acos)
    return 2 * angle

def rotate(point, angle):
    """
    Rotate a set of points by a given angle.

    Args:
        point (numpy.ndarray): Points to rotate.
        angle (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: Rotated points.
    """
    px, py = point[:, 0], point[:, 1]
    rotated = np.empty((point.shape[0], 2))
    rotated[:, 0] = math.cos(angle) * px - math.sin(angle) * py
    rotated[:, 1] = math.sin(angle) * px + math.cos(angle) * py
    return rotated

def to_global_frame(cones, state):
    """
    Convert cone positions to the global frame based on the car's state.

    Args:
        cones (numpy.ndarray): Cone positions.
        state (fsds.CarState): State of the car.

    Returns:
        numpy.ndarray: Cone positions in the global frame.
    """
    rotated = rotate(cones, math.radians(get_orientation(state.kinematics_estimated.orientation)))
    rotated[:, 0] = rotated[:, 0] - state.kinematics_estimated.position.y_val
    rotated[:, 1] = rotated[:, 1] + state.kinematics_estimated.position.x_val
    rotated[:, 2] = cones[:, 2] + state[-1]
    return rotated
