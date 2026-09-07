import numpy as np
from config import PCU_WEIGHTS

class VehicleDetector:
    """
    Nhận dạng và tính tải trọng quy đổi PCU (Chương 5)
    """
    def __init__(self, model_path=None):
        self.model_path = model_path

    def detect_and_count_pcu(self, frame):
        # Trả về giá trị mẫu khi chưa cắm mô hình thật
        dummy_detections = [
            {"label": "car", "confidence": 0.88, "box": [100, 150, 200, 250]},
            {"label": "motorcycle", "confidence": 0.92, "box": [50, 60, 90, 120]}
        ]
        total_pcu = sum(PCU_WEIGHTS.get(d["label"], 1.0) for d in dummy_detections)
        return total_pcu, dummy_detections
