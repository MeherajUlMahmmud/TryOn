import math
from typing import NamedTuple, Tuple, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from utils import Helper


class PoseDetector:
    """
    A class to perform human pose detection using Mediapipe.

    Attributes:
        static_image_mode (bool): Whether to treat the input images as static.
        model_complexity (int): The complexity of the pose estimation model.
        smooth_landmarks (bool): Whether to apply smoothing to landmarks.
        enable_segmentation (bool): Whether to enable segmentation.
        smooth_segmentation (bool): Whether to apply smoothing to segmentation masks.
        min_detection_confidence (float): Minimum confidence value for detecting a pose.
        min_tracking_confidence (float): Minimum confidence value for tracking landmarks.

    Methods:
        find_pose(img: np.ndarray, draw: bool = True) -> np.ndarray:
            Detects pose landmarks in the input image and optionally draws them.

        find_position(img: np.ndarray, draw: bool = True, bbox_with_hands: bool = False) -> Tuple[List[Tuple[int, int, int]], dict]:
            Finds the position of landmarks in the input image and optionally draws a bounding box.

        find_distance(p1: Tuple[int, int], p2: Tuple[int, int], img: Optional[np.ndarray] = None, color: Tuple[int, int, int] = (255, 0, 255), scale: int = 5) -> Tuple[float, Optional[np.ndarray], Tuple[int, int, int, int, int, int]]:
            Calculates the distance between two points and optionally draws it on the image.

        find_angle(p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int], img: Optional[np.ndarray] = None, color: Tuple[int, int, int] = (255, 0, 255), scale: int = 5) -> Tuple[float, Optional[np.ndarray]]:
            Calculates the angle between three points and optionally draws it on the image.

        angle_check(my_angle: float, target_angle: float, offset: float = 20) -> bool:
            Checks if an angle is within a specified range.
    """

    def __init__(
            self,
            static_image_mode: bool = False,
            model_complexity: int = 1,
            smooth_landmarks: bool = True,
            enable_segmentation: bool = False,
            smooth_segmentation: bool = True,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5,
    ):
        """Initializes the PoseDetector with the specified options."""
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.results: Optional[NamedTuple] = None
        self.landmarks_list: List[Tuple[int, int, int]] = []
        self.bbox_info: dict = {}

    def find_pose(self, img: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Detects pose landmarks in the input image.

        Args:
            img (np.ndarray): The input image.
            draw (bool): Whether to draw the landmarks on the image.

        Returns:
            np.ndarray: The image with landmarks drawn (if draw=True).
        """
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_RGB)
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img, self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return img

    def find_position(
            self,
            img: np.ndarray,
            draw: bool = True,
            bbox_with_hands: bool = False) -> Tuple[List[Tuple[int, int, int]], dict]:
        """
        Finds the position of landmarks in the input image.

        Args:
            img (np.ndarray): The input image.
            draw (bool): Whether to draw the bounding box and center on the image.
            bbox_with_hands (bool): Whether to include hands in the bounding box.

        Returns:
            Tuple[List[Tuple[int, int, int]], dict]: A list of landmarks and a dictionary with bounding box information.
        """
        self.landmarks_list.clear()
        self.bbox_info.clear()

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.landmarks_list.append((cx, cy, cz))

            ad = abs(self.landmarks_list[12][0] - self.landmarks_list[11][0]) // 2
            x1, x2 = (
                (self.landmarks_list[16][0] - ad, self.landmarks_list[15][0] + ad)
                if bbox_with_hands else
                (self.landmarks_list[12][0] - ad, self.landmarks_list[11][0] + ad)
            )
            y1 = self.landmarks_list[1][1] - ad
            y2 = self.landmarks_list[29][1] + ad

            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
            self.bbox_info = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, bbox, (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.landmarks_list, self.bbox_info

    @staticmethod
    def find_distance(
            p1: Tuple[int, int], p2: Tuple[int, int],
            img: Optional[np.ndarray] = None,
            color: Tuple[int, int, int] = (255, 0, 255),
            scale: int = 5) -> Tuple[float, Optional[np.ndarray], Tuple[int, int, int, int, int, int]]:
        """
        Calculates the distance between two points.

        Args:
            p1 (Tuple[int, int]): First point (x1, y1).
            p2 (Tuple[int, int]): Second point (x2, y2).
            img (Optional[np.ndarray]): Image to draw the distance on.
            color (Tuple[int, int, int]): Color of the line and circles.
            scale (int): Scale of the circles.

        Returns:
            Tuple[float, Optional[np.ndarray], Tuple[int, int, int, int, int, int]]:
                The distance between the points, the image with the distance drawn, and the coordinates of the points.
        """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length, img, info

    @staticmethod
    def find_angle(
            p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int],
            img: Optional[np.ndarray] = None,
            color: Tuple[int, int, int] = (255, 0, 255),
            scale: int = 5
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Calculates the angle between three points.

        Args:
            p1 (Tuple[int, int]): First point (x1, y1).
            p2 (Tuple[int, int]): Second point (x2, y2).
            p3 (Tuple[int, int]): Third point (x3, y3).
            img (Optional[np.ndarray]): Image to draw the angle on.
            color (Tuple[int, int, int]): Color of the lines and circles.
            scale (int): Scale of the circles.

        Returns:
            Tuple[float, Optional[np.ndarray]]: The angle between the points and the image with the angle drawn.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        angle = angle + 360 if angle < 0 else angle

        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), max(1, scale // 5))
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), max(1, scale // 5))
            for point in [p1, p2, p3]:
                cv2.circle(img, point, scale, color, cv2.FILLED)
                cv2.circle(img, point, scale + 5, color, max(1, scale // 5))
            Helper.put_text_on_frame(
                img, str(int(angle)), (x2 - 50, y2 + 50),
                color=color, scale=max(1, scale // 5)
            )

        return angle, img

    @staticmethod
    def angle_check(my_angle: float, target_angle: float, offset: float = 20) -> bool:
        """
        Checks if a given angle is within a specified range.

        Args:
            my_angle (float): The angle to check.
            target_angle (float): The target angle.
            offset (float): The allowed deviation from the target angle.

        Returns:
            bool: True if the angle is within the range, False otherwise.
        """
        return target_angle - offset < my_angle < target_angle + offset
