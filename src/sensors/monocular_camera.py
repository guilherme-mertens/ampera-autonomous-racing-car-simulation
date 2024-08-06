import torch
import numpy as np
import cv2


class MonocularCamera:
    """
    MonocularCamera class for detecting cones using a monocular camera and a YOLO model.

    Attributes:
        model (torch.nn.Module): YOLO model for object detection.
    """

    def __init__(self, model_path):
        """
        Initializes the MonocularCamera object with the given model path.

        Args:
            model_path (str): Path to the YOLO model.
        """
        self.model = self.init_model(model_path)

    @staticmethod
    def init_model(yolo_path):
        """
        Initializes the YOLO model.

        Args:
            yolo_path (str): Path to the YOLO model.

        Returns:
            torch.nn.Module: Initialized YOLO model.
        """
        yolo_source = r"C:\Users\victo/.cache\torch\hub\ultralytics_yolov5_master"
        yolo_model = torch.hub.load(yolo_source, 'custom', path=yolo_path, source="local")
        return yolo_model

    def detect_cones(self, image):
        """
        Detects cones in the given image.

        Args:
            image (np.ndarray): Image to process.

        Returns:
            torch.Tensor: Detection results.
        """
        return self.model(image)

    @staticmethod
    def get_center_boxes(results):
        """
        Generates the boxes' center coordinates.

        Args:
            results (np.ndarray): Detection results containing bounding boxes.

        Returns:
            np.ndarray: Array of center coordinates with confidence and class.
        """
        center_boxes = np.empty((results.shape[0], 4))
        center_boxes[:, 0] = (results[:, 0] + results[:, 2]) / 2
        center_boxes[:, 1] = (results[:, 1] + results[:, 3]) / 2
        center_boxes[:, 2] = results[:, -2]  # confidence
        center_boxes[:, 3] = results[:, -1]  # class
        return center_boxes

    def get_cones_position_in_image(self, image):
        """
        Detects and returns the positions of cones in the image.

        Args:
            image (np.ndarray): Image to process.

        Returns:
            np.ndarray: Array of detected cones' positions.
        """
        results = self.detect_cones(image).xyxy[0].cpu().numpy()
        results = results[np.where(results[:, 4] > 0.75)]
        return self.get_center_boxes(results)
