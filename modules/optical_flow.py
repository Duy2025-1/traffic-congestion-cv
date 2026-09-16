import cv2
import numpy as np

class MotionEstimator:
    """
    Ước lượng vận tốc dòng xe qua Optical Flow (Chương 3)
    """
    def __init__(self, noise_threshold=1.0):
        self.prev_gray = None
        self.noise_threshold = noise_threshold

    def estimate_speed(self, current_frame):
        if len(current_frame.shape) == 3:
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = current_frame

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, np.zeros_like(gray, dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.prev_gray = gray

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Lọc bỏ nhiễu rung lắc nhỏ
        valid_magnitudes = magnitude[magnitude >= self.noise_threshold]
        
        # Tính vận tốc trung bình của toàn bộ dòng xe đang lưu thông trên làn đường
        if len(valid_magnitudes) > 0:
            avg_speed = float(np.mean(valid_magnitudes))
        else:
            avg_speed = 0.0
            
        return avg_speed, magnitude
