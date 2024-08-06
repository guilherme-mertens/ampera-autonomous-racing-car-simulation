import numpy as np
from math import atan2, pi, cos, sin

class StanleyController:
    """
    Stanley Controller for vehicle path tracking.

    Attributes:
        trajectory (numpy.ndarray): The planned trajectory for the vehicle.
        k (float): Cross track distance gain.
        front_axle (float): Distance from car center to its front axle.
        point_index (int): Index of the reference point in trajectory.
        cross_track_distance (float): Cross track distance from the front axle to the reference point.
        heading_error (float): Heading error component.
        cross_track_error (float): Cross track error component.
        car_velocity (float): Current car velocity.
        car_pos_x (float): Current car X position.
        car_pos_y (float): Current car Y position.
        car_yaw (float): Current car yaw angle.
    """
    
    def __init__(self, k, front_axle):
        self.trajectory = None
        self.k = k
        self.front_axle = front_axle
        self.point_index = None
        self.cross_track_distance = 0
        self.heading_error = 0
        self.cross_track_error = 0
        self.car_velocity = 0
        self.car_pos_x = 0
        self.car_pos_y = 0
        self.car_yaw = 0

    def get_cross_track(self):
        """
        Calculate the cross track distance.
        """
        difference = np.sqrt(np.sum(np.square(self.trajectory[:, :2] - np.array([self.car_pos_x, self.car_pos_y])), axis=1))
        self.point_index = difference.argmin()
        self.check_point_location()
        ref_state = self.trajectory[self.point_index]
        self.cross_track_distance = sin((-1) * ref_state[-1]) * (self.car_pos_x - ref_state[0]) + cos((-1) * ref_state[-1]) * (self.car_pos_y - ref_state[1])

    def check_point_location(self):
        """
        Update the reference point index if necessary.
        """
        point = self.trajectory[self.point_index]
        if cos(self.car_yaw) * point[0] - sin(self.car_yaw) * point[1] < cos(self.car_yaw) * self.car_pos_x - sin(self.car_yaw) * self.car_pos_y:
            self.point_index += 1

    def get_heading_error(self):
        """
        Calculate the heading error.
        """
        self.heading_error = self.car_yaw - self.trajectory[self.point_index][-1]

    def get_steering(self):
        """
        Calculate the steering angle using the Stanley method.

        Returns:
            float: Steering angle.
        """
        self.get_cross_track()
        self.get_heading_error()
        if self.heading_error > pi:
            self.heading_error += -2 * pi
        if self.heading_error < -pi:
            self.heading_error += 2 * pi
        self.cross_track_error = atan2(self.k * self.cross_track_distance, self.car_velocity)
        steering = self.cross_track_error + self.heading_error
        if steering > pi / 4:
            steering = pi / 4
        if steering < -pi / 4:
            steering = -pi / 4
        return steering

    def update_state(self, pos_x, pos_y, yaw, vel):
        """
        Update the vehicle state.

        Args:
            pos_x (float): X position of the vehicle.
            pos_y (float): Y position of the vehicle.
            yaw (float): Yaw angle of the vehicle.
            vel (float): Velocity of the vehicle.
        """
        self.car_yaw = yaw
        self.car_velocity = vel
        self.car_pos_x = pos_x + cos(self.car_yaw) * self.front_axle
        self.car_pos_y = pos_y + sin(self.car_yaw) * self.front_axle