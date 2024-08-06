import numpy as np
import cv2
import math


class IPM:
    def __init__(self):
        """
        Initializes the IPM (Inverse Perspective Mapping) class.
        """
        self.lidar_cones = None
        self.image_cones = None
        self.ipm_cones = None
        self.image_x_gain = 4 / 200
        self.image_y_gain = 8 / 200
        self.image_y_bias = 0.5

    def ipm(self):
        """
        Applies the perspective transformation to the image cones.

        Returns:
            np.ndarray: Transformed cones in the IPM perspective.
        """
        # Define the coordinates of the corners of the perspective transformation
        dir_sup = (629, 390)
        esq_sup = (451, 390)
        dir_inf = (983, 545)
        esq_inf = (97, 545)

        esq_sup_out = (431, 450)
        dir_sup_out = (649, 450)
        esq_inf_out = (451, 700)
        dir_inf_out = (629, 700)

        # Vertices coordinates in the source image
        source_points = np.array([
            [esq_sup[0], esq_sup[1]],
            [dir_sup[0], dir_sup[1]],
            [esq_inf[0], esq_inf[1]],
            [dir_inf[0], dir_inf[1]]
        ], dtype=np.float32)

        # Vertices coordinates in the destination image
        target_points = np.array([
            [esq_sup_out[0], esq_sup_out[1]],
            [dir_sup_out[0], dir_sup_out[1]],
            [esq_inf_out[0], esq_inf_out[1]],
            [dir_inf_out[0], dir_inf_out[1]]
        ], dtype=np.float32)

        # Compute the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(source_points, target_points)
        cones_a = np.array([self.image_cones[:, :2]])

        # Apply the perspective transformation to the cones
        result = cv2.perspectiveTransform(cones_a, matrix)

        return result[0]

    def get_ipm_cones(self):
        """
        Computes the IPM cones and updates the `ipm_cones` attribute with the transformed coordinates.

        This method applies the perspective transformation and adjusts the cone coordinates based on 
        image gains and bias.
        """
        cones_transformed = self.ipm()
        cones_mean = np.concatenate((cones_transformed, self.image_cones[:, 2:]), axis=1)
        cones_offset = np.empty(cones_mean.shape)
        cones_offset[:, 0] = cones_mean[:, 0] - 540
        cones_offset[:, 1] = 720 - cones_mean[:, 1]
        cones_offset[:, 2] = cones_mean[:, 2]
        cones_offset[:, 3] = cones_mean[:, 3]

        # Apply gains and bias to the coordinates
        cones_offset[:, 0] = cones_offset[:, 0] * self.image_x_gain
        cones_offset[:, 1] = cones_offset[:, 1] * self.image_y_gain + self.image_y_bias
        self.ipm_cones = cones_offset

    def tint_cones(self):
        """
        Matches and colors the IPM cones based on LIDAR data.

        Returns:
            np.ndarray: LIDAR cones with assigned colors based on proximity to IPM cones.
        """
        aux_ipm_cones = np.repeat(self.ipm_cones, self.lidar_cones.shape[0], axis=0)
        aux_lidar_cones = np.tile(self.lidar_cones, (self.ipm_cones.shape[0], 1))
        distances = np.sqrt(
            np.square(aux_ipm_cones[:, 0] - aux_lidar_cones[:, 0]) +
            np.square(aux_ipm_cones[:, 1] - aux_lidar_cones[:, 1])
        )
        possible_matches = np.where(distances < 1.5)[0]

        colored_lidar_cones = np.append(
            aux_lidar_cones[possible_matches],
            np.reshape(aux_ipm_cones[possible_matches][:, -1],
                       newshape=(aux_ipm_cones[possible_matches].shape[0], 1)),
            axis=1
        )
        difference_check = np.append(
            colored_lidar_cones,
            np.reshape(distances[possible_matches],
                       newshape=(aux_ipm_cones[possible_matches].shape[0], 1)),
            axis=1
        )
        sorted_by_dif = difference_check[np.argsort(difference_check[:, -1])]
        uniques = np.unique(sorted_by_dif, axis=0)
        _, single_cones = np.unique(uniques[:, 0], return_index=True)

        return uniques[:, :3][single_cones]

    def tint_cones_with_confidence(self):
        """
        Matches and colors the IPM cones with confidence scores based on LIDAR data.

        Returns:
            np.ndarray: LIDAR cones with assigned colors and confidence scores based on proximity to IPM cones.
        """
        aux_ipm_cones = np.repeat(self.ipm_cones, self.lidar_cones.shape[0], axis=0)
        aux_lidar_cones = np.tile(self.lidar_cones, (self.ipm_cones.shape[0], 1))
        distances = np.sqrt(
            np.square(aux_ipm_cones[:, 0] - aux_lidar_cones[:, 0]) +
            np.square(aux_ipm_cones[:, 1] - aux_lidar_cones[:, 1])
        )
        possible_matches = np.where(distances < 1.5)[0]

        colored_lidar_cones = np.append(
            aux_lidar_cones[possible_matches],
            np.reshape(aux_ipm_cones[possible_matches][:, 2:4],
                       newshape=(aux_ipm_cones[possible_matches].shape[0], 2)),
            axis=1
        )
        difference_check = np.append(
            colored_lidar_cones,
            np.reshape(distances[possible_matches],
                       newshape=(aux_ipm_cones[possible_matches].shape[0], 1)),
            axis=1
        )
        sorted_by_dif = difference_check[np.argsort(difference_check[:, -1])]
        uniques = np.unique(sorted_by_dif, axis=0)
        _, single_cones = np.unique(uniques[:, 0], return_index=True)

        return uniques[:, :4][single_cones]

    def rotate(self, points, angle):
        """
        Rotates a set of points by a given angle.

        Args:
            points (np.ndarray): Array of points to rotate.
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: Rotated points.
        """
        px, py = points[:, 0], points[:, 1]
        rotated = np.empty((points.shape[0], 2))
        rotated[:, 0] = math.cos(angle) * px - math.sin(angle) * py
        rotated[:, 1] = math.sin(angle) * px + math.cos(angle) * py
        return rotated

    def upload_cones(self, image_cones, lidar_data):
        """
        Uploads image cones and LIDAR data to the class and performs rotation on LIDAR data.

        Args:
            image_cones (np.ndarray): Array of cones detected in the image.
            lidar_data (np.ndarray): Array of LIDAR cone data.
        """
        self.image_cones = image_cones
        lidar_cones = self.rotate(lidar_data, math.pi / 2)
        self.lidar_cones = lidar_cones

    def get_cones(self):
        """
        Retrieves the colored LIDAR cones by applying IPM and tinting them.

        Returns:
            np.ndarray: LIDAR cones with colors assigned based on proximity to IPM cones.
        """
        self.get_ipm_cones()
        colored_lidar_cones = self.tint_cones()
        return colored_lidar_cones

    def get_cones_with_confidence(self):
        """
        Retrieves the colored LIDAR cones with confidence scores by applying IPM and tinting them.

        Returns:
            np.ndarray: LIDAR cones with colors and confidence scores assigned based on proximity to IPM cones.
        """
        self.get_ipm_cones()
        return self.tint_cones_with_confidence()
