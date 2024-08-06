import numpy as np
from scipy import interpolate
import shapely.geometry as geom
import math


class PathPlanning:
    """
    PathPlanningLap class for planning a single lap path.

    Attributes:
        cones (np.ndarray): Array of cone positions.
        trajectory (np.ndarray): Array of trajectory points.
        state (list): Current state of the vehicle.
    """

    def __init__(self):
        """
        Initializes the PathPlanningLap object with default state.
        """
        self.cones = None
        self.trajectory = None
        self.state = [[0, 0], 0]  # [[x, y], orientation]

    def interpolate(self, path):
        """
        Interpolates the path to create a smooth trajectory.

        Args:
            path (np.ndarray): Array of path points.

        Returns:
            np.ndarray: Array of interpolated trajectory points.
        """
        tck, u = interpolate.splprep(path.T, s=0)
        unew = np.arange(0, 1.01, 0.01)
        out = interpolate.splev(unew, tck)
        return np.vstack(out).T

    def sort_path(self, path_points):
        """
        Sorts the path points to create a coherent trajectory.

        Args:
            path_points (np.ndarray): Array of unsorted path points.

        Returns:
            np.ndarray: Array of sorted path points.
        """
        car_position = np.array(self.state[0])
        sorted_path = []

        # Find the closest point to the car and start sorting from there
        distances = np.linalg.norm(path_points - car_position, axis=1)
        start_idx = np.argmin(distances)

        sorted_path.append(path_points[start_idx])
        path_points = np.delete(path_points, start_idx, axis=0)

        while len(path_points) > 0:
            last_point = sorted_path[-1]
            distances = np.linalg.norm(path_points - last_point, axis=1)
            next_idx = np.argmin(distances)
            sorted_path.append(path_points[next_idx])
            path_points = np.delete(path_points, next_idx, axis=0)

        return np.array(sorted_path)

    def process(self, cones):
        """
        Processes the cones to generate a trajectory.

        Args:
            cones (np.ndarray): Array of cone positions.
        """
        self.cones = cones
        path = self.sort_path(self.cones)
        self.trajectory = self.interpolate(path)

    def update(self, data):
        """
        Updates the path planning with new data.

        Args:
            data: New data containing cone positions.
        """
        positions = data[:, :2]
        colors = data[:, 2]
        self.cones = np.column_stack((positions, colors))

    def get_trajectory(self):
        """
        Returns the current trajectory.

        Returns:
            np.ndarray: Array of trajectory points.
        """
        return self.trajectory
