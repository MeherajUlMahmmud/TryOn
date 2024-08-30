import os
from typing import Union, Any, List, Tuple

import cv2
import numpy as np

from constants import Constant
from logger import setup_logger
from pose_detector import PoseDetector
from utils import Helper

APP_NAME = Constant.app_name
T_SHIRT_PATH = Constant.t_shirt_path

logger = setup_logger()

selected_t_shirt: Union[np.ndarray, Any] = None
t_shirts: List[str] = os.listdir(T_SHIRT_PATH)
loaded_t_shirts: List[np.ndarray] = []
selected_t_shirt_idx: Union[int, Any] = None  # To track the selected T-shirt

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Camera not opened")
    logger.info("Exiting...")
    exit()

logger.info("Camera opened")

frame_shape = (0, 0)
detector = PoseDetector()
run: bool = True

logger.info("Loading t-shirt images")
# Load all T-shirt images
for t_shirt in t_shirts:
    t_shirt_image: np.ndarray = cv2.imread(os.path.join(T_SHIRT_PATH, t_shirt))

    # Check if the image is loaded correctly
    if t_shirt_image is None:
        logger.error(f"Failed to load {t_shirt}. Skipping this file.")
        continue

    loaded_t_shirts.append(t_shirt_image)
logger.info(f"Loaded {len(loaded_t_shirts)} t-shirts images")


def show_t_shirt_list(frame: np.ndarray, t_shirts: List[np.ndarray], selected_idx: int) -> np.ndarray:
    """
    Displays a list of T-shirt icons at the bottom of the given frame, highlighting the selected one.

    Args:
        frame (np.ndarray): The frame on which to display the T-shirt icons.
        t_shirts (List[np.ndarray]): A list of T-shirt images to display as icons.
        selected_idx (int): The index of the currently selected T-shirt.

    Returns:
        np.ndarray: The frame with T-shirt icons displayed at the bottom.
    """
    icon_size = 50  # Size of each T-shirt icon (width and height)
    spacing = 10  # Space between each icon
    frame_height, frame_width = frame.shape[:2]  # Get frame dimensions

    # Calculate the total width of all icons combined, including spacing
    total_width = len(t_shirts) * (icon_size + spacing) - spacing

    # Calculate x offset to center the icons horizontally on the frame
    x_offset = (frame_width - total_width) // 2
    # Calculate y offset to position the icons near the bottom of the frame
    y_offset = frame_height - icon_size - spacing

    for idx, t_shirt in enumerate(t_shirts):
        # Resize the T-shirt image to the icon size
        t_shirt_icon = cv2.resize(t_shirt, (icon_size, icon_size))

        # Place the resized icon onto the frame at the calculated position
        frame[y_offset:y_offset + icon_size, x_offset:x_offset + icon_size] = t_shirt_icon

        # Determine the border color: red if selected, green otherwise
        border_color = (0, 0, 255) if idx == selected_idx else (0, 255, 0)
        # Draw a rectangle around the icon to highlight it
        cv2.rectangle(
            frame,
            (x_offset, y_offset),
            (x_offset + icon_size, y_offset + icon_size),
            border_color,
            2,
        )

        # Update the x offset for the next icon
        x_offset += icon_size + spacing

    return frame


def click_event(event, x, y, flags, param):
    """
    Handles mouse click events to select or deselect a T-shirt icon displayed on the frame.

    Args:
        event: The type of mouse event (e.g., left button click).
        x: The x-coordinate of the mouse event.
        y: The y-coordinate of the mouse event.
        flags: Any relevant flags passed by OpenCV.
        param: Additional parameters (unused in this function).
    """
    global selected_t_shirt_idx, selected_t_shirt

    if event == cv2.EVENT_LBUTTONDOWN:
        icon_size = 50  # Size of the T-shirt icon
        spacing = 10  # Space between each icon
        total_width = len(loaded_t_shirts) * (icon_size + spacing) - spacing

        # Calculate x offset to center the icons horizontally on the frame
        x_offset = (frame_shape[1] - total_width) // 2
        # Calculate y offset to position the icons near the bottom of the frame
        y_offset = frame_shape[0] - icon_size - spacing

        for idx in range(len(loaded_t_shirts)):
            x1 = x_offset + idx * (icon_size + spacing)
            y1 = y_offset
            x2 = x1 + icon_size
            y2 = y1 + icon_size

            # Check if the click is within the bounds of a T-shirt icon
            if x1 <= x <= x2 and y1 <= y <= y2:
                if selected_t_shirt_idx == idx:
                    # Deselect the T-shirt if the same icon is clicked again
                    selected_t_shirt_idx = None
                    selected_t_shirt = None
                else:
                    # Select the clicked T-shirt
                    selected_t_shirt_idx = idx
                    selected_t_shirt = loaded_t_shirts[idx]
                break


def get_landmark(landmarks_list: List[Tuple[int, int, int]], idx):
    """
    Retrieves the x and y coordinates of a specific landmark from the landmarks list.

    Args:
        landmarks_list (List[List[int]]): A list of landmarks, where each landmark is a list containing x and y coordinates.
        idx (int): The index of the landmark to retrieve.

    Returns:
        tuple: A tuple (x, y) representing the coordinates of the landmark if the index is valid.
        None: If the index is out of range or the landmarks list is empty.
    """
    if 0 <= idx < len(landmarks_list):
        return landmarks_list[idx][0], landmarks_list[idx][1]
    return None


# Set the mouse callback function
cv2.namedWindow(APP_NAME)
cv2.setMouseCallback(APP_NAME, click_event)

# Main loop
while run:
    ret, frame = cap.read()
    frame_shape = frame.shape
    frame = detector.find_pose(frame)
    landmarks_list, bbox_info = detector.find_position(
        frame,
        bbox_with_hands=False,
    )

    if bbox_info:
        center = bbox_info["center"]
        cv2.circle(frame, center, 5, (255, 0, 255), cv2.FILLED)

    # Display the T-shirt icons on the frame
    frame = show_t_shirt_list(frame, loaded_t_shirts, selected_t_shirt_idx)

    # Get post data from landmarks
    left_shoulder = get_landmark(landmarks_list, 12)
    right_shoulder = get_landmark(landmarks_list, 13)
    left_elbow = get_landmark(landmarks_list, 14)
    right_elbow = get_landmark(landmarks_list, 15)
    left_hip = get_landmark(landmarks_list, 24)
    right_hip = get_landmark(landmarks_list, 25)

    print("Left Shoulder: ", left_shoulder)
    print("Right Shoulder: ", right_shoulder)
    print("Left Elbow: ", left_elbow)
    print("Right Elbow: ", right_elbow)
    print("Left Hip: ", left_hip)
    print("Right Hip: ", right_hip)

    if left_shoulder:
        left_shoulder_x, left_shoulder_y = left_shoulder[0], left_shoulder[1]
        Helper.put_text_on_frame(
            frame,
            f"Left Shoulder: ({left_shoulder_x}, {left_shoulder_y})",
            (left_shoulder_x, left_shoulder_y - 20),
        )

    if right_shoulder:
        right_shoulder_x, right_shoulder_y = right_shoulder[0], right_shoulder[1]
        Helper.put_text_on_frame(
            frame,
            f"Right Shoulder: ({right_shoulder_x}, {right_shoulder_y})",
            (right_shoulder_x, right_shoulder_y - 20),
        )

    if left_elbow:
        left_elbow_x, left_elbow_y = left_elbow[0], left_elbow[1]
        Helper.put_text_on_frame(
            frame,
            f"Left Elbow: ({left_elbow_x}, {left_elbow_y})",
            (left_elbow_x, left_elbow_y - 20),
        )

    if right_elbow:
        right_elbow_x, right_elbow_y = right_elbow[0], right_elbow[1]
        Helper.put_text_on_frame(
            frame,
            f"Right Elbow: ({right_elbow_x}, {right_elbow_y})",
            (right_elbow_x, right_elbow_y - 20),
        )

    if left_hip:
        left_hip_x, left_hip_y = left_hip[0], left_hip[1]
        Helper.put_text_on_frame(
            frame,
            f"Left Hip: ({left_hip_x}, {left_hip_y})",
            (left_hip_x, left_hip_y - 20),
        )

    if right_hip:
        right_hip_x, right_hip_y = right_hip[0], right_hip[1]
        Helper.put_text_on_frame(
            frame,
            f"Right Hip: ({right_hip_x}, {right_hip_y})",
            (right_hip_x, right_hip_y - 20),
        )

    if selected_t_shirt is not None:
        # Determine where you want to place the selected T-shirt on the frame
        shirt_height, shirt_width = selected_t_shirt.shape[:2]

        if left_shoulder and right_shoulder:
            shoulder_width = right_shoulder[0] - left_shoulder[0]

            if left_hip and right_hip:
                hip_width = right_hip[0] - left_hip[0]
                # Resize the T-shirt to fit between the shoulders and hips
                width = min(shoulder_width, hip_width)
                # Maintain aspect ratio
                height = int((width / shirt_width) * shirt_height)

                # Resize the T-shirt image
                resized_t_shirt = cv2.resize(selected_t_shirt, (width, height))

                # Position the T-shirt between the shoulders and hips
                x_pos = left_shoulder[0]
                y_pos = left_shoulder[0]

                # Ensure the resized T-shirt fits within the frame
                if y_pos + height > frame.shape[0]:
                    y_pos = frame.shape[0] - height

                if x_pos + width > frame.shape[1]:
                    x_pos = frame.shape[1] - width

                    # Overlay the T-shirt on the frame
                frame[y_pos:y_pos + height, x_pos:x_pos + width] = resized_t_shirt
        else:
            # Center horizontally
            x_pos = (frame_shape[1] - shirt_width) // 2
            y_pos = 50  # Position vertically, adjust as needed
            frame[y_pos:y_pos + shirt_height, x_pos:x_pos + shirt_width] = selected_t_shirt

    cv2.imshow(APP_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        run = False
        break

cap.release()
cv2.destroyAllWindows()
