import cv2

from logger import setup_logger

logger = setup_logger()


class Helper:

    @staticmethod
    def put_text_on_frame(frame, text, position, font=cv2.FONT_HERSHEY_PLAIN, scale=1, color=(0, 255, 0), thickness=2):
        cv2.putText(frame, text, position, font, scale, color, thickness)
