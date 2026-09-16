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
    # Bổ sung class 1 (bicycle/xe hai bánh nhỏ) quy đổi về nhóm xe máy (0.33 PCU)
    COCO_VEHICLE_MAP = {
        1: "motorcycle",  # Xe đạp / xe điện / moped hai bánh
        2: "car",         # Ô tô con / Taxi
        3: "motorcycle",  # Xe máy / môtô
        5: "bus",         # Xe buýt
        7: "truck"        # Xe tải / xe container
    }

    # Bảng màu hiển thị trực quan (BGR)
    COLOR_MAP = {
        "motorcycle": (255, 255, 0),  # Xanh ngọc (Cyan)
        "car":        (0, 255, 0),    # Xanh lá (Green)
        "bus":        (0, 165, 255),  # Cam (Orange)
        "truck":      (0, 0, 255)     # Đỏ (Red - xe tải lớn)
    }

    def __init__(
        self,
        model_path=None,
        conf_thresh=0.22,
        iou_thresh=0.45,
        imgsz=1280,
        high_accuracy=False,
        device=None
    ):
        """
        Khởi tạo VehicleDetector tối ưu hóa độ chính xác và phạm vi nhận diện.
        
        Args:
            model_path: Đường dẫn tới file trọng số (.pt) hoặc tên mô hình (mặc định ưu tiên yolov8s.pt > yolov8n.pt)
            conf_thresh: Ngưỡng tin cậy tối ưu (mặc định 0.22 giúp phát hiện đầy đủ xe ở xa/bị che khuất)
            iou_thresh: Ngưỡng NMS IoU Threshold
            imgsz: Độ phân giải suy luận (mặc định 1280 giữ trọn vẹn chi tiết trên video 1080p)
            high_accuracy: Bật quét 2 tầng (Two-pass Sliced Detection) cho vùng xe ở xa chân trời
            device: Thiết bị chạy mô hình ('cpu', 'cuda', hoặc None để tự động)
        """
        # Tự động chọn mô hình tối ưu nhất có sẵn trên máy (ưu tiên yolov8n.pt chuẩn bài toán TV5)
        if model_path is None:
            if os.path.exists("yolov8n.pt"):
                model_path = "yolov8n.pt"
            elif os.path.exists("yolov8s.pt"):
                model_path = "yolov8s.pt"
            elif os.path.exists("weights/best.pt"):
                model_path = "weights/best.pt"
            else:
                model_path = "yolov8n.pt"

        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.high_accuracy = high_accuracy
        self.device = device
        self.model = None
        self.is_yolo_loaded = False
        self.class_map = self.COCO_VEHICLE_MAP
        self.infer_classes = list(self.COCO_VEHICLE_MAP.keys())

        # Lưu trữ kết quả gần nhất phục vụ vẽ trực quan và thống kê
        self.last_pcu = 0.0
        self.last_counts = {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0}
        self.last_detections = []

        self._load_model()

    def _load_model(self):
        """Nạp mô hình YOLOv8 qua Ultralytics với cơ chế fallback tự động."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.is_yolo_loaded = True

            # Tự động đồng bộ class_map theo cấu trúc nhãn của mô hình
            if hasattr(self.model, "names") and self.model.names:
                names = self.model.names
                if len(names) == 4 and "motorcycle" in names.values():
                    # Mô hình fine-tuned 4 lớp chuẩn PCU
                    self.class_map = {idx: name for idx, name in names.items()}
                    self.infer_classes = list(self.class_map.keys())
                else:
                    # Mô hình COCO chuẩn
                    self.class_map = self.COCO_VEHICLE_MAP
                    self.infer_classes = list(self.COCO_VEHICLE_MAP.keys())

            print(f"[VehicleDetector] Da nap thanh cong YOLO model: {self.model_path} (imgsz={self.imgsz}, conf={self.conf_thresh})")
        except Exception as e:
            self.is_yolo_loaded = False
            self.model = None
            print(f"[VehicleDetector CANH BAO] Khong the nap YOLO ({e}). Che do Fallback/Dummy duoc kich hoat.")

    def detect_and_count_pcu(self, frame, return_counts=False, conf=None, iou=None, imgsz=None):
        """
        Nhận dạng phương tiện trên toàn khung hình và tính toán tổng chỉ số PCU.
        
        Args:
            frame: Khung hình video (BGR)
            return_counts: Nếu True trả về (total_pcu, counts, detections)
            conf: Tùy chỉnh confidence threshold cho frame hiện tại
            iou: Tùy chỉnh iou threshold cho frame hiện tại
            imgsz: Tùy chỉnh resolution inference
        """
        if frame is None:
            if return_counts:
                return 0.0, {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0}, []
            return 0.0, []

        conf_threshold = conf if conf is not None else self.conf_thresh
        iou_threshold = iou if iou is not None else self.iou_thresh
        infer_imgsz = imgsz if imgsz is not None else self.imgsz

        counts = {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0}
        detections = []
        total_pcu = 0.0

        if self.is_yolo_loaded and self.model is not None:
            try:
                # 1. Quét toàn khung hình ở độ phân giải cao
                results = self.model(
                    frame,
                    classes=self.infer_classes,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=infer_imgsz,
                    device=self.device,
                    verbose=False
                )[0]

                raw_boxes = []
                raw_scores = []
                raw_labels = []

                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    label = self.class_map.get(cls_id, None)
                    if label:
                        raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
                        raw_scores.append(confidence)
                        raw_labels.append(label)

                # 2. Quét tầng 2 tăng cường vùng đường xa (Two-pass Sliced Detection) nếu bật high_accuracy
                if self.high_accuracy:
                    h, w = frame.shape[:2]
                    y_start, y_end = int(0.20 * h), int(0.65 * h)
                    crop_distant = frame[y_start:y_end, :]
                    
                    results_distant = self.model(
                        crop_distant,
                        classes=self.infer_classes,
                        conf=max(0.18, conf_threshold - 0.04),
                        iou=iou_threshold,
                        imgsz=infer_imgsz,
                        device=self.device,
                        verbose=False
                    )[0]

                    for box in results_distant.boxes:
                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        label = self.class_map.get(cls_id, None)
                        if label:
                            # Bù lại tọa độ gốc của frame
                            real_y1 = y1 + y_start
                            real_y2 = y2 + y_start
                            raw_boxes.append([x1, real_y1, x2 - x1, real_y2 - real_y1])
                            raw_scores.append(confidence)
                            raw_labels.append(label)

                # Áp dụng Per-Class NMS (Non-Maximum Suppression theo từng lớp phương tiện)
                # Đảm bảo xe máy đi liền kề hoặc bị xe tải/ô tô che khuất một phần không bị ức chế nhầm
                if raw_boxes:
                    for target_lbl in ["motorcycle", "car", "bus", "truck"]:
                        lbl_indices = [i for i, l in enumerate(raw_labels) if l == target_lbl]
                        if not lbl_indices:
                            continue
                        sub_boxes = [raw_boxes[i] for i in lbl_indices]
                        sub_scores = [raw_scores[i] for i in lbl_indices]
                        indices = cv2.dnn.NMSBoxes(sub_boxes, sub_scores, conf_threshold, iou_threshold)
                        if len(indices) > 0:
                            for k in indices.flatten():
                                orig_idx = lbl_indices[k]
                                bx, by, bw, bh = raw_boxes[orig_idx]
                                conf_val = raw_scores[orig_idx]
                                counts[target_lbl] += 1
                                detections.append({
                                    "label": target_lbl,
                                    "confidence": round(conf_val, 3),
                                    "box": [bx, by, bx + bw, by + bh]
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
        Hỗ trợ vẽ thích ứng cho xe ở xa và chống tràn biên màn hình.
        
        Args:
            frame: Khung hình video gốc
            detections: Danh sách bounding box (mặc định lấy self.last_detections)
            draw_hud: Có vẽ bảng thông tin tổng hợp PCU và số lượng xe ở góc màn hình hay không
        """
        if frame is None:
            return None

        disp_frame = frame.copy()
        h, w = disp_frame.shape[:2]
        target_detections = detections if detections is not None else self.last_detections

        for d in target_detections:
            x1, y1, x2, y2 = d["box"]
            # Kẹp tọa độ trong phạm vi khung hình
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))

            box_w = x2 - x1
            box_h = y2 - y1

            lbl = d["label"]
            conf = d.get("confidence", 0.0)
            color = self.COLOR_MAP.get(lbl, (255, 255, 255))

            # 1. Vẽ bounding box thích ứng (xe xa/nhỏ dùng nét 1px thanh thoát)
            line_thickness = 1 if (box_h < 40 or box_w < 40) else 2
            cv2.rectangle(disp_frame, (x1, y1), (x2, y2), color, line_thickness)

            # 2. Vẽ nhãn và độ tin cậy với font_scale thích ứng
            font_scale = 0.35 if (box_h < 40 or box_w < 40) else 0.48
            text_thickness = 1
            label_str = f"{lbl} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(label_str, font, font_scale, text_thickness)

            # Đảm bảo nhãn không tràn viền trên hoặc phải
            text_y = max(y1, text_h + 4)
            text_x = min(x1, w - text_w - 6)

            cv2.rectangle(
                disp_frame,
                (text_x, text_y - text_h - 4),
                (text_x + text_w + 4, text_y + baseline - 2),
                color,
                cv2.FILLED
            )
            cv2.putText(
                disp_frame,
                label_str,
                (text_x + 2, text_y - 2),
                font,
                font_scale,
                (0, 0, 0),
                text_thickness,
                cv2.LINE_AA
            )

        # 3. Vẽ bảng tóm tắt HUD nếu được kích hoạt
        if draw_hud:
            self._draw_pcu_hud(disp_frame)

        return disp_frame

    def _draw_pcu_hud(self, frame):
        """Vẽ bảng tóm tắt số lượng từng phương tiện và tổng PCU ở góc trên bên phải."""
        h, w = frame.shape[:2]
        box_w, box_h = 230, 145
        pad_x, pad_y = w - box_w - 20, 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (pad_x, pad_y), (pad_x + box_w, pad_y + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (pad_x, pad_y), (pad_x + box_w, pad_y + box_h), (255, 255, 255), 1)

        cv2.putText(frame, "[TV5] DONG XE & PCU", (pad_x + 10, pad_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        c = self.last_counts
        lines = [
            (f"Xe may (x0.33): {c.get('motorcycle', 0)}", self.COLOR_MAP["motorcycle"]),
            (f"O to con (x1.0): {c.get('car', 0)}", self.COLOR_MAP["car"]),
            (f"Xe buyt  (x2.5): {c.get('bus', 0)}", self.COLOR_MAP["bus"]),
            (f"Xe tai   (x3.0): {c.get('truck', 0)}", self.COLOR_MAP["truck"]),
            (f"Tong PCU: {self.last_pcu:.2f}", (0, 255, 255))
        ]

        curr_y = pad_y + 42
        for text, color in lines:
            cv2.putText(frame, text, (pad_x + 10, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            curr_y += 20

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
