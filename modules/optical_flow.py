import cv2
import numpy as np

class MotionEstimator:
    """
    Ước lượng vận tốc dòng xe qua Optical Flow (Chương 3)
    """
    def __init__(self):
        self.prev_gray = None

    def estimate_speed(self, current_frame):
        if len(current_frame.shape) == 3:
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = current_frame

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, np.zeros_like(gray)

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.prev_gray = gray

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_speed = float(np.mean(magnitude))
        return avg_speed, magnitude
