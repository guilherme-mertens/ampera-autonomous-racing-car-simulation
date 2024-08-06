from threading import Thread
import time
import warnings
import numpy as np
import math
import os
import sys
from utils.pipeline_utils import rotate, get_orientation
from sensors.lidar import Lidar
from sensors.monocular_camera import MonocularCamera
from perception.ipm import IPM
from perception.slam import SLAM
from controllers.stanley_controller import StanleyController
from planning.path_planning import PathPlanning
import fsds

warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore")


class Systems:
    """
    Main class to manage autonomous vehicle systems including perception, mapping, path planning, and control.
    """
    def __init__(self, client_perception, client_control, car_controls):
        self.stanley_controller = StanleyController(5, 0.4)
        self.camera = MonocularCamera(os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python", "yolov5s_weights_best_result.pt"))
        self.slam = SLAM(cluster_range=1.1)
        self.lidar = Lidar(20)
        self.ipm = IPM()
        self.n = 0
        self.client_perception = client_perception
        self.client_control = client_control
        self.perception_state = None
        self.trajectory = None
        self.car_controls = car_controls
        self.current_cones = None
        self.doMapping = False
        self.localCones, self.globalCones = None, None
        self.planner = PathPlanning()
        self.last_local_state = None
        self.doPlanning = False
        self.doSave = False
        self.doControl = False
        self.car_state = self.client_control.getCarState()
        self.car_controls.throttle = 0.01
        self.frameindex = 0
        self.superviseSteering = np.empty((0, 1))

    def perception(self):
        """Thread function for perception using camera and LiDAR."""
        while True:
            time.sleep(0.02)
            try:
                [image] = self.client_perception.simGetImages(
                    [fsds.ImageRequest(camera_name="cam3", image_type=fsds.ImageType.Scene, pixels_as_float=False, compress=False)],
                    vehicle_name='FSCar'
                )
                lidar_data = self.client_perception.getLidarData(lidar_name='Lidar1')
                self.perception_state = self.client_perception.getCarState()
                img1d = np.frombuffer(image.image_data_uint8, dtype=np.uint8)
                img_rgb = np.flip(img1d.reshape(image.height, image.width, 3), axis=2)
                image_cones = self.camera.get_cones_position_in_image(img_rgb)
                lidar_cones = self.lidar.LiDAR_Detect(lidar_data)
                if lidar_cones.size != 0:
                    self.ipm.upload_cones(image_cones, lidar_cones)
                    self.current_cones = self.ipm.get_cones()
                    self.doMapping = True
            except Exception as e:
                print(f"Error in perception: {e}")

    def mapping(self):
        """Thread function for mapping using SLAM."""
        while True:
            time.sleep(0.1)
            if self.doMapping:
                self.doMapping = False
                self.last_local_state = self.perception_state
                self.slam.UpdateGlobalCones(self.current_cones, self.last_local_state)
                self.slam.AcessLocalView(self.car_state, self.car_controls.steering, supervise_index=self.frameindex)
                self.superviseSteering = np.r_[self.superviseSteering, [[self.car_controls.steering]]]
                np.savetxt(os.path.join("Supervisor", "GlobalCones.csv"), self.slam.globalCones, delimiter=",")
                self.frameindex += 1
                self.doPlanning = True

    def path_planning(self):
        """Thread function for path planning."""
        while True:
            time.sleep(0.1)
            if self.doPlanning:
                car_state = [
                    [self.car_state.kinematics_estimated.position.x_val, self.car_state.kinematics_estimated.position.y_val],
                    math.radians(get_orientation(self.car_state.kinematics_estimated.orientation))
                ]
                self.planner.update_state(car_state, self.slam.localCones)
                try:
                    self.trajectory = self.planner.get_trajectory()
                    self.stanley_controller.trajectory = self.trajectory
                except Exception as error:
                    print(f"Error in path planning: {error}")
                self.doSave = True
                self.doControl = True

    def control(self):
        """Thread function for vehicle control using Stanley controller."""
        while True:
            time.sleep(0.01)
            if self.doControl:
                try:
                    self.car_state = self.client_control.getCarState()
                    velocity = np.linalg.norm(self.car_state.kinematics_estimated.linear_velocity.to_numpy_array()[:2])
                    pos_x = self.car_state.kinematics_estimated.position.x_val
                    pos_y = self.car_state.kinematics_estimated.position.y_val
                    yaw = math.radians(get_orientation(self.car_state.kinematics_estimated.orientation))
                    self.stanley_controller.update_state(pos_x, pos_y, yaw, velocity)
                    self.car_controls.steering = math.tan(self.stanley_controller.get_steering())
                    self.client_control.setCarControls(self.car_controls)
                except Exception as e:
                    print(f"Error in control: {e}")

    def run(self):
        """Start all system threads."""
        threads = [
            Thread(target=self.perception),
            Thread(target=self.mapping),
            Thread(target=self.path_planning),
            Thread(target=self.control)
        ]
        for t in threads:
            t.start()


def main():
    fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
    sys.path.insert(0, fsds_lib_path)
    client_control = fsds.FSDSClient()
    client_perception = fsds.FSDSClient()
    client_perception.confirmConnection()
    client_control.confirmConnection()
    client_control.enableApiControl(True)
    car_controls = fsds.CarControls()
    car = Systems(client_perception, client_control, car_controls)
    car.run()


if __name__ == "__main__":
    main()
