import numpy as np
import math
from scipy.spatial import cKDTree
from perception.filters import Filtros

class SLAM:
    """
    SLAM (Simultaneous Localization and Mapping) for cone-based localization.

    Attributes:
        position (numpy.ndarray): Current position of the car.
        headingAngle (float): Current heading angle of the car.
        clusterRange (float): Range for clustering cones.
        globalCones (numpy.ndarray): Global cones mapping.
        localCones (numpy.ndarray): Local cones mapping.
        filter (Filtros): Filter object for cone processing.
    """

    def __init__(self, cluster_range: float):
        self.position = None
        self.headingAngle = None
        self.clusterRange = cluster_range
        self.globalCones = np.empty((0, 3))
        self.localCones = np.empty((0, 3))
        self.filter = Filtros()

    def average_point(self, v1, v2):
        """
        Calculate the average point between two points.

        Args:
            v1 (numpy.ndarray): First point.
            v2 (numpy.ndarray): Second point.

        Returns:
            numpy.ndarray: Average point.
        """
        vx = 0.5 * (v1[0] + v2[0])
        vy = 0.5 * (v1[1] + v2[1])
        return np.array([vx, vy])
    
    def rotate(self, point, angle):
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

    def rotate_1d(self, point, angle):
        """
        Rotate a single point by a given angle.

        Args:
            point (numpy.ndarray): Point to rotate.
            angle (float): Rotation angle in radians.

        Returns:
            list: Rotated point.
        """
        px, py = point[0], point[1]
        rpx = math.cos(angle) * px - math.sin(angle) * py
        rpy = math.sin(angle) * px + math.cos(angle) * py
        return [rpx, rpy]

    def angle_filter(self, fsds_or):
        """
        Convert orientation to an angle.

        Args:
            fsds_or (fsds.Quaternionr): Quaternion orientation.

        Returns:
            float: Angle in degrees.
        """
        z_or = fsds_or.z_val
        z_orneg = z_or < 0
        z_angle = 2 * math.asin(z_or) * 180 / math.pi
        if z_orneg:
            z_angle += 360
        return z_angle

    def get_car_pose(self, state):
        """
        Get the current pose of the car.

        Args:
            state (fsds.CarState): State of the car.

        Returns:
            tuple: Position and angle of the car.
        """
        position = np.array([state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val])
        angle = self.angle_filter(state.kinematics_estimated.orientation)
        return position, angle

    def cluster(self, cones_info: np.ndarray):
        """
        Cluster the cones based on their positions.

        Args:
            cones_info (numpy.ndarray): Cones information.

        Returns:
            numpy.ndarray: Clustered cones.
        """
        ConesPosition = cones_info[:, :2]
        ConesColor = cones_info[:, -1]
        tree = cKDTree(ConesPosition)
        RowsToFuse = np.array(list(tree.query_pairs(r=self.clusterRange)))
        del_col = list()

        for rtf in RowsToFuse:
            x, y = self.average_point(ConesPosition[rtf[0]], ConesPosition[rtf[1]])
            np.put(ConesPosition, 2 * rtf[0], x)
            np.put(ConesPosition, 2 * rtf[0] + 1, y)
            newColor = ConesColor[rtf[1]]
            np.put(ConesColor, rtf[0], newColor)
            del_col.append(rtf[1])

        ConesPosition = np.delete(ConesPosition, del_col, axis=0)
        ConesColor = np.delete(ConesColor, del_col, axis=0)
        return np.c_[ConesPosition, ConesColor] 

    def update_global_cones(self, colored_cones, state):
        """
        Update the global cones mapping with detected cones and current car state.

        Args:
            colored_cones (numpy.ndarray): Detected cones with perception system.
            state (fsds.CarState): State of the car.
        """
        self.position, self.headingAngle = self.get_car_pose(state)
        colored_cones[:, [0, 1]] = colored_cones[:, [1, 0]]
        colored_cones[:, 1] *= -1
        rotatedCones = self.rotate(colored_cones[:, :2], self.headingAngle * np.pi / 180) + self.position
        rotatedCones = np.c_[rotatedCones, colored_cones[:, -1]]
        self.globalCones = np.r_[self.globalCones, rotatedCones]
        self.globalCones = self.cluster(self.globalCones)
    
    def access_local_view(self, state, steering_value, supervise_index=None, view_range=[20, 20]):
        """
        Update the local cones mapping with the global cones mapping.

        Args:
            state (fsds.CarState): State of the car.
            steering_value (float): Steering value of the car.
            supervise_index (int, optional): Index for supervision. Defaults to None.
            view_range (list, optional): Cut-off range for local cones. Defaults to [20, 20].
        """
        position, headingAngle = self.get_car_pose(state)
        relativeCones = self.rotate(self.globalCones[:, :2], -headingAngle * np.pi / 180)
        relativeCones = np.c_[relativeCones, self.globalCones[:, -1]]
        position = self.rotate_1d(position, -headingAngle * np.pi / 180)
        relativeCones[:, 0] -= position[0]
        relativeCones[:, 1] -= position[1]
        inRangeConesIndex = np.where(
            (abs(relativeCones[:, 1]) < view_range[0]) &
            (relativeCones[:, 0] < view_range[1]) &
            (relativeCones[:, 0] > 0)
        )
        relativeCones = relativeCones[inRangeConesIndex]

        try:
            relativeCones[:, :2] = self.rotate(relativeCones[:, :2], np.pi / 2)
            relativeCones = self.filter.remove_far_cones(relativeCones, steering_value)
            relativeCones = self.filter.find_current_lane(relativeCones, steering_value)
            relativeCones = self.filter.no_blue(relativeCones)
            relativeCones = self.filter.no_yellow(relativeCones)
            relativeCones[:, :2] = self.rotate(relativeCones[:, :2], -np.pi / 2)
        except Exception as e:
            print(f"Error in filtering cones: {e}")

        relativeCones[:, 0] += position[0]
        relativeCones[:, 1] += position[1]
        relativeCones[:, :2] = self.rotate(relativeCones[:, :2], headingAngle * np.pi / 180)
        self.localCones = relativeCones

    def access_local_view_heuristic(self, state, steering_value, supervise_index=None, view_range=[20, 20]):
        """
        Update the local cones mapping with the global cones mapping using heuristic.

        Args:
            state (fsds.CarState): State of the car.
            steering_value (float): Steering value of the car.
            supervise_index (int, optional): Index for supervision. Defaults to None.
            view_range (list, optional): Cut-off range for local cones. Defaults to [20, 20].
        """
        position, headingAngle = self.get_car_pose(state)
        relativeCones = self.rotate(self.globalCones[:, :2], -headingAngle * np.pi / 180)
        relativeCones = np.c_[relativeCones, self.globalCones[:, -1]]
        position = self.rotate_1d(position, -headingAngle * np.pi / 180)
        relativeCones[:, 0] -= position[0]
        relativeCones[:, 1] -= position[1]
        inRangeConesIndex = np.where(
            (abs(relativeCones[:, 1]) < view_range[0]) &
            (relativeCones[:, 0] < view_range[1]) &
            (relativeCones[:, 0] > 0)
        )
        relativeCones = relativeCones[inRangeConesIndex]

        try:
            relativeCones[:, :2] = self.rotate(relativeCones[:, :2], np.pi / 2)
            relativeCones = self.filter.remove_far_cones(relativeCones, steering_value)
            relativeCones = self.filter.find_current_lane(relativeCones, steering_value)
            relativeCones = self.filter.no_blue(relativeCones)
            relativeCones = self.filter.no_yellow(relativeCones)
            relativeCones = self.filter.correct_cones(relativeCones)
            relativeCones[:, :2] = self.rotate(relativeCones[:, :2], -np.pi / 2)
        except Exception as e:
            print(f"Error in heuristic filtering cones: {e}")

        relativeCones[:, 0] += position[0]
        relativeCones[:, 1] += position[1]
        relativeCones[:, :2] = self.rotate(relativeCones[:, :2], headingAngle * np.pi / 180)
        self.localCones = relativeCones

    def update_and_view(self, state, colored_cones, view_range):
        """
        Update the global mapping and extract local mapping.

        Args:
            state (fsds.CarState): State of the car.
            colored_cones (numpy.ndarray): Detected cones with perception system.
            view_range (list): Cut-off range for local cones.
        """
        self.update_global_cones(colored_cones, state)
        self.access_local_view(state, view_range)
