import math
from typing import NamedTuple

import cv2
import mediapipe as mp

from utils import Helper


class PoseDetector:

    def __init__(
            self, static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
    ):
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

        self.results: NamedTuple = None
        self.landmarks_list = []
        self.bbox_info = {}

    def find_pose(self, img, draw=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_RGB)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(
                    img, self.results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                )
        return img

    def find_position(self, img, draw=True, bbox_with_hands=False):
        self.landmarks_list = []
        self.bbox_info = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.landmarks_list.append([cx, cy, cz])

            # Bounding Box
            ad = abs(self.landmarks_list[12][0] -
                     self.landmarks_list[11][0]) // 2
            if bbox_with_hands:
                x1 = self.landmarks_list[16][0] - ad
                x2 = self.landmarks_list[15][0] + ad
            else:
                x1 = self.landmarks_list[12][0] - ad
                x2 = self.landmarks_list[11][0] + ad

            y2 = self.landmarks_list[29][1] + ad
            y1 = self.landmarks_list[1][1] - ad
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + bbox[3] // 2

            self.bbox_info = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, bbox, (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.landmarks_list, self.bboxInfo

    @staticmethod
    def find_distance(p1, p2, img=None, color=(255, 0, 255), scale=5):
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
    def find_angle(p1, p2, p3, img=None, color=(255, 0, 255), scale=5):
        # Get the landmarks
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Calculate the Angle
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        if angle < 0:
            angle += 360

        # Draw
        if img is not None:
            cv2.line(
                img,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                max(1, scale // 5),
            )
            cv2.line(
                img,
                (x3, y3),
                (x2, y2),
                (255, 255, 255),
                max(1, scale // 5),
            )
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x1, y1), scale + 5, color, max(1, scale // 5))
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale + 5, color, max(1, scale // 5))
            cv2.circle(img, (x3, y3), scale, color, cv2.FILLED)
            cv2.circle(img, (x3, y3), scale + 5, color, max(1, scale // 5))
            Helper.put_text_on_frame(
                img,
                str(int(angle)),
                (x2 - 50, y2 + 50),
                color=color,
                scale=max(1, scale // 5),
            )
        return angle, img

    @staticmethod
    def angle_check(my_angle, target_angle, offset=20):
        return target_angle - offset < my_angle < target_angle + offset
