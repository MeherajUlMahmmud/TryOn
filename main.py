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

logger.info("====================================================")
logger.info("Starting up...")

selected_t_shirt: Union[np.ndarray, Any] = None
t_shirt_paths: List[str] = os.listdir(T_SHIRT_PATH)
loaded_t_shirts: List[np.ndarray] = []
selected_t_shirt_idx: Union[int, Any] = None  # To track the selected T-shirt
frame_shape = (0, 0)


def load_t_shirt_images():
    global loaded_t_shirts

    # Load all T-shirt images
    logger.info("Loading t-shirt images")
    for t_shirt in t_shirt_paths:
        t_shirt_image: np.ndarray = cv2.imread(
            os.path.join(T_SHIRT_PATH, t_shirt), cv2.IMREAD_UNCHANGED)

        # Check if the image is loaded correctly
        if t_shirt_image is None:
            logger.error(f"Failed to load {t_shirt}. Skipping this file.")
            continue

        loaded_t_shirts.append(t_shirt_image)
    logger.info(f"Loaded {len(loaded_t_shirts)} t-shirts images")


def show_t_shirt_list(
        frame: np.ndarray,
        t_shirts: List[np.ndarray],
        selected_idx: int,
        position: str = "bottom-center"
) -> np.ndarray:
    """
    Displays a list of T-shirt icons at the specified position on the frame, highlighting the selected one.

    Args:
        frame (np.ndarray): The frame on which to display the T-shirt icons.
        t_shirts (List[np.ndarray]): A list of T-shirt images to display as icons.
        selected_idx (int): The index of the currently selected T-shirt.
        position (str): The position on the screen where the icons should be displayed.
                        Options: "top-left", "top-center", "top-right",
                                 "bottom-left", "bottom-center", "bottom-right",
                                 "center-left", "center-center", "center-right".

    Returns:
        np.ndarray: The frame with T-shirt icons displayed at the specified position.
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

    # # Calculate x offset based on the position
    # if "left" in position:
    #     x_offset = spacing
    # elif "center" in position:
    #     x_offset = (frame_width - total_width) // 2
    # else:  # "right"
    #     x_offset = frame_width - total_width - spacing
    #
    # # Calculate y offset based on the position
    # if "top" in position:
    #     y_offset = spacing
    # elif "center" in position:
    #     y_offset = (frame_height - icon_size) // 2
    # else:  # "bottom"
    #     y_offset = frame_height - icon_size - spacing

    for idx, t_shirt_img in enumerate(t_shirts):
        # Resize the T-shirt image to the icon size
        t_shirt_icon = cv2.resize(t_shirt_img, (icon_size, icon_size))

        # Check if the resized image has 4 channels (RGBA)
        if t_shirt_icon.shape[2] == 4:
            # Convert RGBA to RGB by removing the alpha channel
            t_shirt_icon = cv2.cvtColor(t_shirt_icon, cv2.COLOR_RGBA2RGB)

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


def click_event(event: int, x: int, y: int, flags, param):
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


def get_landmark(landmarks: List[Tuple[int, int, int]], idx):
    """
    Retrieves the x and y coordinates of a specific landmark from the landmarks list.

    Args:
        landmarks (List[List[int]]): A list of landmarks, where each landmark is a list containing x and y coordinates.
        idx (int): The index of the landmark to retrieve.

    Returns:
        tuple: A tuple (x, y) representing the coordinates of the landmark if the index is valid.
        None: If the index is out of range or the landmarks list is empty.
    """
    if 0 <= idx < len(landmarks):
        return landmarks[idx][0], landmarks[idx][1], landmarks[idx][2]
    return None


# Set the mouse callback function
cv2.namedWindow(APP_NAME)
cv2.setMouseCallback(APP_NAME, click_event)


def main():
    global frame_shape

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Camera not opened")
        logger.info("Exiting...")
        exit()
    logger.info("Camera opened")

    load_t_shirt_images()

    detector = PoseDetector()
    run: bool = True

    # Main loop
    while run:
        success, frame = cap.read()
        if not success:
            break

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
        left_shoulder = get_landmark(landmarks_list, 11)
        right_shoulder = get_landmark(landmarks_list, 12)
        left_elbow = get_landmark(landmarks_list, 13)
        right_elbow = get_landmark(landmarks_list, 14)
        left_hip = get_landmark(landmarks_list, 23)
        right_hip = get_landmark(landmarks_list, 24)

        # print("Left Shoulder: ", left_shoulder)
        # print("Right Shoulder: ", right_shoulder)
        # print("Left Elbow: ", left_elbow)
        # print("Right Elbow: ", right_elbow)
        # print("Left Hip: ", left_hip)
        # print("Right Hip: ", right_hip)

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
            shirt_height, shirt_width = selected_t_shirt.shape[:2]

            if left_shoulder and right_shoulder and left_hip and right_hip:
                shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])
                hip_distance = abs(left_hip[0] - right_hip[0])
                shoulder_hip_distance = abs(left_shoulder[1] - left_hip[1])

                # The width of the shirt should cover the greater distance between the shoulders or the hips
                width_of_shirt = max(shoulder_distance, hip_distance)

                #  The height of the shirt should cover the distance from shoulder to hip
                height_of_shirt = shoulder_hip_distance

                # Resize the T-shirt image to fit the calculated width and height
                resized_t_shirt = cv2.resize(
                    selected_t_shirt,
                    (width_of_shirt, height_of_shirt),
                )

                try:
                    # frame = Helper.overlay_image(frame, resized_t_shirt, right_shoulder[:2])
                    frame = Helper.overlay_image(
                        frame, resized_t_shirt,
                        (right_shoulder[0], right_shoulder[1]),
                    )
                except Exception as e:
                    logger.error(f"Error; {e}")
                    pass

        cv2.imshow(APP_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Pressed q, closing window...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
