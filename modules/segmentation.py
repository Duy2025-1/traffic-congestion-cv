import cv2
import numpy as np

class RoadSegmenter:
    """
    Phân đoạn mặt đường và phát hiện diện tích chiếm dụng (Chương 4)
    """
    def __init__(self, history=500, var_threshold=16, detect_shadows=True):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )

    def extract_occupancy(self, frame, roi_mask=None):
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        if roi_mask is not None:
            cleaned_mask = cv2.bitwise_and(cleaned_mask, cleaned_mask, mask=roi_mask)
            total_pixels = cv2.countNonZero(roi_mask)
        else:
            total_pixels = frame.shape[0] * frame.shape[1]
            
        vehicle_pixels = cv2.countNonZero(cleaned_mask)
        occupancy_ratio = float(vehicle_pixels / total_pixels) if total_pixels > 0 else 0.0
        return occupancy_ratio, cleaned_mask
