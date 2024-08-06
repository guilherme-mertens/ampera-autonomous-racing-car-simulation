from scipy.spatial import Delaunay
import numpy as np
import itertools
from scipy import interpolate
from utils.pipeline_utils import rotate, get_orientation
import shapely.geometry as geom
import math


class PathPlanning:
    """
    PathPlanning class for planning paths using Delaunay triangulation and optimization.

    Attributes:
        cones (np.ndarray): Array of cone positions.
        trajectory (np.ndarray): Array of trajectory points.
        state (list): Current state of the vehicle.
    """

    def __init__(self):
        """
        Initializes the PathPlanning object with default state.
        """
        self.cones = None
        self.trajectory = None
        self.state = [[0, 0], 0]  # [[x, y], orientation]

    def delaunay(self):
        """
        Creates the Delaunay triangulation.

        Returns:
            scipy.spatial.Delaunay: Delaunay triangulation object.
        """
        centers = self.cones[:, :2]
        return Delaunay(centers)

    def find_path(self, triangulation):
        """
        Creates the path by connecting blue and yellow cones.

        Args:
            triangulation (np.ndarray): Array of Delaunay triangles.

        Returns:
            np.ndarray: Array of path center points.
        """
        arr_centers = self.cones
        path_center = []

        for triangle in triangulation:
            first_vertex, second_vertex, third_vertex = triangle

            if arr_centers[first_vertex][2] == 0:  # blue cone
                if arr_centers[second_vertex][2] == 2:  # yellow cone
                    path_x = (arr_centers[first_vertex][0] + arr_centers[second_vertex][0]) / 2
                    path_y = (arr_centers[first_vertex][1] + arr_centers[second_vertex][1]) / 2
                    path_center.append([path_x, path_y])
                if arr_centers[third_vertex][2] == 2:  # yellow cone
                    path_x = (arr_centers[first_vertex][0] + arr_centers[third_vertex][0]) / 2
                    path_y = (arr_centers[first_vertex][1] + arr_centers[third_vertex][1]) / 2
                    path_center.append([path_x, path_y])

            if arr_centers[first_vertex][2] == 2:  # yellow cone
                if arr_centers[second_vertex][2] == 0:  # blue cone
                    path_x = (arr_centers[first_vertex][0] + arr_centers[second_vertex][0]) / 2
                    path_y = (arr_centers[first_vertex][1] + arr_centers[second_vertex][1]) / 2
                    path_center.append([path_x, path_y])
                if arr_centers[third_vertex][2] == 0:  # blue cone
                    path_x = (arr_centers[first_vertex][0] + arr_centers[third_vertex][0]) / 2
                    path_y = (arr_centers[first_vertex][1] + arr_centers[third_vertex][1]) / 2
                    path_center.append([path_x, path_y])
        return np.array(path_center)

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

    def process(self, cones):
        """
        Processes the cones to generate a trajectory.

        Args:
            cones (np.ndarray): Array of cone positions.
        """
        self.cones = cones
        triangulation = self.delaunay().simplices
        path = self.find_path(triangulation)
        sorted_path = self.sort_path(path)
        self.trajectory = self.interpolate(sorted_path)

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
