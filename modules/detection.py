"""
MODULE NHẬN DẠNG PHƯƠNG TIỆN & QUY ĐỔI TẢI TRỌNG PCU (CHƯƠNG 5)
Phụ trách: TV5
Học phần: Xử lý ảnh và Thị giác máy tính (121036) - UTH
Dự án: Hệ thống Giám sát và Đánh giá Ùn tắc Giao thông Đô thị

Nhiệm vụ:
1. Phân loại 4 nhóm phương tiện theo chuẩn COCO:
   - Xe máy (motorcycle - class 3)
   - Ô tô con / Taxi (car - class 2)
   - Xe buýt (bus - class 5)
   - Xe tải (truck - class 7)
2. Quy đổi sang Đơn vị Xe con Tiêu chuẩn (Passenger Car Unit - PCU):
   - Xe máy: 0.33 PCU (3 xe máy ~ 1 ô tô con)
   - Ô tô con: 1.0 PCU (Đơn vị cơ sở)
   - Xe buýt: 2.5 PCU
   - Xe tải: 3.0 PCU (Cản trở lưu thông và tải trọng lớn)
3. Khảo sát tham số (Parameter Sweep) conf_threshold & iou_threshold để tối ưu
   độ phân tách giữa xe buýt và xe tải thùng dài.
"""

import os
import cv2
import numpy as np
from config import PCU_WEIGHTS


class VehicleDetector:
    """
    Bộ nhận dạng và phân loại phương tiện giao thông tích hợp YOLOv8.
    Quy đổi lưu lượng thực tế sang hệ số tải trọng PCU cho bộ tính toán TCI (TV1).
    """

    # Bảng ánh xạ nhãn COCO chuẩn sang 4 nhóm phương tiện bài toán yêu cầu
    COCO_VEHICLE_MAP = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    # Bảng màu hiển thị trực quan (BGR)
    COLOR_MAP = {
        "motorcycle": (255, 255, 0),  # Xanh ngọc (Cyan)
        "car":        (0, 255, 0),    # Xanh lá (Green)
        "bus":        (0, 165, 255),  # Cam (Orange)
        "truck":      (0, 0, 255)     # Đỏ (Red - xe tải lớn)
    }

    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.4, iou_thresh=0.45, device=None):
        """
        Khởi tạo VehicleDetector.
        Args:
            model_path: Đường dẫn tới file trọng số (.pt) hoặc tên mô hình ('yolov8n.pt')
            conf_thresh: Ngưỡng tin cậy (Confidence Threshold) mặc định
            iou_thresh: Ngưỡng NMS IoU Threshold
            device: Thiết bị chạy mô hình ('cpu', 'cuda', hoặc None để tự động)
        """
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.model = None
        self.is_yolo_loaded = False
        
        # Thống kê lần nhận dạng gần nhất
        self.last_counts = {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0}
        self.last_pcu = 0.0
        self.last_detections = []

        self._load_model()

    def _load_model(self):
        """Nạp mô hình YOLOv8 qua Ultralytics với cơ chế fallback tự động."""
        try:
            from ultralytics import YOLO
            # Nếu model_path không tồn tại cục bộ, ultralytics tự động tải về
            self.model = YOLO(self.model_path if self.model_path else "yolov8n.pt")
            self.is_yolo_loaded = True
            print(f"[VehicleDetector] Da nap thanh cong YOLO model: {self.model_path}")
        except Exception as e:
            self.is_yolo_loaded = False
            self.model = None
            print(f"[VehicleDetector CANH BAO] Khong the nap YOLO ({e}). Che do Fallback/Dummy duoc kich hoat.")

    def detect_and_count_pcu(self, frame, return_counts=False, conf=None, iou=None):
        """
        Nhận dạng phương tiện và tính toán tổng chỉ số PCU.
        
        Args:
            frame: Khung hình video (BGR)
            return_counts: Nếu True trả về (total_pcu, counts, detections). 
                           Mặc định False trả về (total_pcu, detections) để đồng bộ 100% với main.py.
            conf: Tùy chỉnh confidence threshold cho frame hiện tại
            iou: Tùy chỉnh iou threshold cho frame hiện tại
            
        Returns:
            total_pcu: Tổng tải trọng quy đổi (float)
            detections: Danh sách dict [{'label', 'confidence', 'box': [x1, y1, x2, y2]}]
            (optional) counts: Dict số lượng từng loại xe
        """
        if frame is None:
            if return_counts:
                return 0.0, {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0}, []
            return 0.0, []

        conf_threshold = conf if conf is not None else self.conf_thresh
        iou_threshold = iou if iou is not None else self.iou_thresh

        counts = {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0}
        detections = []
        total_pcu = 0.0

        if self.is_yolo_loaded and self.model is not None:
            try:
                # Chạy inference chỉ với các lớp phương tiện
                results = self.model(
                    frame,
                    classes=list(self.COCO_VEHICLE_MAP.keys()),
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=self.device,
                    verbose=False
                )[0]

                boxes = results.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                    label = self.COCO_VEHICLE_MAP.get(cls_id, None)
                    if label in counts:
                        counts[label] += 1
                        detections.append({
                            "label": label,
                            "confidence": round(confidence, 3),
                            "box": [x1, y1, x2, y2]
                        })

            except Exception as e:
                print(f"[VehicleDetector LOI] Inference error: {e}. Su dung du lieu du phong.")
                counts, detections = self._get_fallback_detections(frame)
        else:
            counts, detections = self._get_fallback_detections(frame)

        # Tính tổng tải trọng PCU
        for lbl, cnt in counts.items():
            weight = PCU_WEIGHTS.get(lbl, 1.0)
            total_pcu += cnt * weight

        total_pcu = round(total_pcu, 2)
        self.last_counts = counts
        self.last_pcu = total_pcu
        self.last_detections = detections

        if return_counts:
            return total_pcu, counts, detections
        return total_pcu, detections

    def _get_fallback_detections(self, frame):
        """Mô phỏng dữ liệu mẫu chuẩn khi chưa có trọng số mô hình hoặc GPU."""
        h, w = frame.shape[:2] if frame is not None else (720, 1280)
        dummy_detections = [
            {"label": "truck", "confidence": 0.91, "box": [int(0.23*w), int(0.14*h), int(0.35*w), int(0.44*h)]},
            {"label": "car", "confidence": 0.88, "box": [int(0.08*w), int(0.21*h), int(0.16*w), int(0.35*h)]},
            {"label": "bus", "confidence": 0.85, "box": [int(0.39*w), int(0.17*h), int(0.48*w), int(0.49*h)]},
            {"label": "motorcycle", "confidence": 0.92, "box": [int(0.04*w), int(0.08*h), int(0.07*w), int(0.17*h)]},
            {"label": "motorcycle", "confidence": 0.89, "box": [int(0.06*w), int(0.10*h), int(0.09*w), int(0.18*h)]}
        ]
        counts = {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0}
        for d in dummy_detections:
            lbl = d["label"]
            if lbl in counts:
                counts[lbl] += 1
        return counts, dummy_detections

    def draw_detections(self, frame, detections=None, draw_hud=True):
        """
        Vẽ bounding box, nhãn và thông tin độ tin cậy trực quan lên frame.
        
        Args:
            frame: Khung hình video gốc
            detections: Danh sách bounding box (mặc định lấy self.last_detections)
            draw_hud: Có vẽ bảng thông tin tổng hợp PCU và số lượng xe ở góc màn hình hay không
        """
        if frame is None:
            return None

        disp_frame = frame.copy()
        target_detections = detections if detections is not None else self.last_detections

        for d in target_detections:
            x1, y1, x2, y2 = d["box"]
            lbl = d["label"]
            conf = d.get("confidence", 0.0)
            color = self.COLOR_MAP.get(lbl, (255, 255, 255))

            # 1. Vẽ bounding box
            cv2.rectangle(disp_frame, (x1, y1), (x2, y2), color, 2)

            # 2. Vẽ nhãn và độ tin cậy với nền màu
            label_str = f"{lbl} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_str, font, font_scale, thickness)

            # Đảm bảo nhãn không bị tràn mép trên
            text_y = max(y1, text_h + 6)
            cv2.rectangle(
                disp_frame,
                (x1, text_y - text_h - 4),
                (x1 + text_w + 4, text_y + baseline - 2),
                color,
                cv2.FILLED
            )
            # Chữ đen tương phản cao trên nền sáng
            cv2.putText(
                disp_frame,
                label_str,
                (x1 + 2, text_y - 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        # 3. Vẽ bảng tóm tắt HUD nếu được kích hoạt
        if draw_hud:
            self._draw_pcu_hud(disp_frame)

        return disp_frame

    def _draw_pcu_hud(self, frame):
        """Vẽ bảng tóm tắt số lượng từng phương tiện và tổng PCU ở góc trên bên phải."""
        h, w = frame.shape[:2]
        pad_x, pad_y = w - 240, 20
        box_w, box_h = 220, 140

        overlay = frame.copy()
        cv2.rectangle(overlay, (pad_x, pad_y), (pad_x + box_w, pad_y + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (pad_x, pad_y), (pad_x + box_w, pad_y + box_h), (255, 255, 255), 1)

        cv2.putText(frame, "[TV5] VEHICLE & PCU", (pad_x + 10, pad_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        c = self.last_counts
        lines = [
            (f"Xe may (x0.33): {c.get('motorcycle', 0)}", self.COLOR_MAP["motorcycle"]),
            (f"O to con (x1.0): {c.get('car', 0)}", self.COLOR_MAP["car"]),
            (f"Xe buyt  (x2.5): {c.get('bus', 0)}", self.COLOR_MAP["bus"]),
            (f"Xe tai   (x3.0): {c.get('truck', 0)}", self.COLOR_MAP["truck"]),
            (f"Tong PCU: {self.last_pcu:.2f}", (0, 255, 255))
        ]

        curr_y = pad_y + 40
        for text, color in lines:
            cv2.putText(frame, text, (pad_x + 10, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            curr_y += 18

    def sweep_parameters(self, frame, conf_thresholds=(0.25, 0.50, 0.75), iou_thresholds=(0.30, 0.45, 0.60)):
        """
        Khảo sát ma trận tham số (Parameter Sweep) giữa conf_threshold và iou_threshold.
        Đặc biệt phục vụ việc phân tích tách biệt giữa xe buýt và xe tải.
        
        Returns:
            results_list: Danh sách dict kết quả khảo sát từng cấu hình
        """
        results_list = []
        for conf in conf_thresholds:
            for iou in iou_thresholds:
                total_pcu, counts, detections = self.detect_and_count_pcu(
                    frame, return_counts=True, conf=conf, iou=iou
                )
                results_list.append({
                    "conf_threshold": conf,
                    "iou_threshold": iou,
                    "motorcycle": counts["motorcycle"],
                    "car": counts["car"],
                    "bus": counts["bus"],
                    "truck": counts["truck"],
                    "total_vehicles": sum(counts.values()),
                    "total_pcu": total_pcu
                })
        return results_list

    def get_summary(self):
        """Trả về thống kê đầy đủ về số lượng và đóng góp PCU."""
        breakdown = {}
        for lbl, cnt in self.last_counts.items():
            weight = PCU_WEIGHTS.get(lbl, 1.0)
            breakdown[lbl] = {
                "count": cnt,
                "weight": weight,
                "subtotal_pcu": round(cnt * weight, 2)
            }
        return {
            "counts": self.last_counts,
            "total_pcu": self.last_pcu,
            "breakdown": breakdown
        }
