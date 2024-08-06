import numpy as np
import math


class Lidar:
    """
    Lidar class for processing lidar data.

    Attributes:
        lidar_range (float): Maximum range of the lidar sensor.
    """

    def __init__(self, lidar_range):
        """
        Initializes the Lidar object with the given range.

        Args:
            lidar_range (float): Maximum range of the lidar sensor.
        """
        self.range = lidar_range

    def pointgroup_to_cone(self, group):
        """
        Converts a group of points to a cone's position by averaging their coordinates.

        Args:
            group (list of lists): List of [x, y] coordinates.

        Returns:
            list: Averaged [x, y] coordinate.
        """
        average_x = sum(point[0] for point in group) / len(group)
        average_y = sum(point[1] for point in group) / len(group)
        return [average_x, average_y]

    @staticmethod
    def distance(x1, y1, x2, y2):
        """
        Calculates the Euclidean distance between two points.

        Args:
            x1 (float): X coordinate of the first point.
            y1 (float): Y coordinate of the first point.
            x2 (float): X coordinate of the second point.
            y2 (float): Y coordinate of the second point.

        Returns:
            float: Euclidean distance between the points.
        """
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def rotate(point, angle):
        """
        Rotates a point by a given angle.

        Args:
            point (np.ndarray): Array of points to rotate.
            angle (float): Angle to rotate the points.

        Returns:
            np.ndarray: Rotated points.
        """
        px, py = point[:, 0], point[:, 1]
        rotated = np.empty((point.shape[0], 2))
        rotated[:, 0] = math.cos(angle) * px - math.sin(angle) * py
        rotated[:, 1] = math.sin(angle) * px + math.cos(angle) * py
        return rotated

    @staticmethod
    def anglefilter(fsds_or):
        """
        Converts an orientation angle to a filtered angle.

        Args:
            fsds_or (float): Orientation angle.

        Returns:
            float: Filtered angle.
        """
        z_or = fsds_or
        z_orneg = z_or < 0
        z_angle = 2 * math.asin(z_or) * 180 / math.pi
        if z_orneg:
            z_angle += 360
        return z_angle

    def lidar_sweep(self, lidardata):
        """
        Processes lidar data to extract point cloud.

        Args:
            lidardata: Lidar data object containing point cloud.

        Returns:
            np.ndarray: Processed point cloud data.
        """
        point_cloud = lidardata.point_cloud
        if len(point_cloud) < 3:
            return np.array([[0, 0, 0]])
        points = np.array(point_cloud, dtype=np.float32)
        return np.reshape(points, (int(points.shape[0] / 3), 3))

    def first_clustering(self, points, plot_limit):
        """
        Clusters points into groups representing cones.

        Args:
            points (np.ndarray): Array of points to cluster.
            plot_limit (float): Limit distance to consider a point within the plot.

        Returns:
            list: List of cone positions.
        """
        current_group = []
        cones = []
        for i in range(1, len(points)):
            distance_to_last_point = self.distance(points[i][0], points[i][1], points[i - 1][0], points[i - 1][1])
            if distance_to_last_point < 0.1:
                current_group.append([points[i][0], points[i][1]])
            else:
                if current_group:
                    cone = self.pointgroup_to_cone(current_group)
                    if self.distance(0, 0, cone[0], cone[1]) < plot_limit:
                        cones.append(cone)
                    current_group = []
        return cones

    def lidar_detect(self, lidardata):
        """
        Detects cones using lidar data.

        Args:
            lidardata: Lidar data object.

        Returns:
            np.ndarray: Array of detected cone positions.
        """
        cones = self.first_clustering(self.lidar_sweep(lidardata), self.range)
        return np.array(cones)
