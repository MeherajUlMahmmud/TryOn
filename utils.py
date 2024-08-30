from typing import List, Tuple

import cv2
import numpy as np

from logger import setup_logger

logger = setup_logger()


class Helper:

    @staticmethod
    def put_text_on_frame(frame, text, position, font=cv2.FONT_HERSHEY_PLAIN, scale=1, color=(0, 255, 0), thickness=2):
        cv2.putText(frame, text, position, font, scale, color, thickness)

    @staticmethod
    def overlay_image(img_back: np.ndarray, img_front: np.ndarray, pos: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Overlays a PNG image with transparency onto another image using alpha blending.

        The function handles out-of-bound positions, including negative coordinates, by cropping
        the overlay image accordingly. Edges are smoothed using alpha blending based on the alpha channel.

        Args:
            img_back (np.ndarray): The background image, expected to be of shape (height, width, 3) or (height, width, 4).
                                   The background can have an optional alpha channel, but only the RGB channels will be used.
            img_front (np.ndarray): The foreground PNG image to overlay, of shape (height, width, 4). The 4th channel
                                    should be the alpha channel representing transparency.
            pos (Tuple[int]): A list specifying the [x, y] coordinates (in pixels) at which to overlay the image.
                             Can be negative or cause the overlay image to go out-of-bounds.

        Returns:
            np.ndarray: A new image with the overlay applied. The shape will be the same as `img_back`.
        """

        hf, wf, _ = img_front.shape
        hb, wb, _ = img_back.shape

        x1, y1 = max(pos[0], 0), max(pos[1], 0)
        x2, y2 = min(pos[0] + wf, wb), min(pos[1] + hf, hb)

        # Calculate overlay start coordinates for the overlay image
        x1_overlay = max(0, -pos[0])
        y1_overlay = max(0, -pos[1])

        # Calculate the dimensions of the slice to overlay
        wf, hf = x2 - x1, y2 - y1

        # If overlay is completely outside background, return original background
        if wf <= 0 or hf <= 0:
            return img_back

        # Extract the alpha channel from the foreground and create the inverse mask
        alpha = img_front[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 3] / 255.0
        inv_alpha = 1.0 - alpha

        # Extract the RGB channels from the foreground
        img_rgb = img_front[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 0:3]

        # # Alpha blend the foreground and background
        # for c in range(0, 3):
        #     img_back[y1:y2, x1:x2, c] = img_back[y1:y2, x1:x2, c] * inv_alpha + img_rgb[:, :, c] * alpha

        # Perform alpha blending using vectorized operations
        img_back[y1:y2, x1:x2, :3] = (inv_alpha[..., None] * img_back[y1:y2, x1:x2, :3] + alpha[..., None] * img_rgb)

        return img_back
