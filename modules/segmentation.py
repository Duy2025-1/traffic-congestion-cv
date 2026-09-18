"""
MODULE PHÂN ĐOẠN & TỶ LỆ CHIẾM DỤNG MẶT ĐƯỜNG THÍCH NGHI (ADAPTIVE ROAD OCCUPANCY)
Phụ trách: TV3 - Nâng cấp toàn diện cho đồ án
Học phần: Xử lý ảnh và Thị giác máy tính (121036) - UTH

Các cải tiến kỹ thuật nổi bật:
1. Thuật toán chiếm dụng lai Hybrid Occupancy Fusion (MOG2 + YOLOv8):
   - Giải quyết triệt để nhược điểm của MOG2 khi xe dừng đỗ lâu (Background Absorption làm tụt occupancy về 0%).
   - Hợp nhất diện tích chuyển động MOG2 và hình chiếu của các phương tiện được phát hiện trong vùng ROI:
     Mask_Occupancy = Mask_MOG2 OR Mask_YOLO_Detections.
2. Tự động ước lượng ROI mặt đường thích nghi (Adaptive Auto-ROI):
   - Tự động nhận diện và tính toán đa giác lòng đường từ vết tích chuyển động và phân bố phương tiện
     đối với bất kỳ video mới hoặc video upload nào mà không cần nhập tay toạ độ.
3. Khử bóng đổ chính xác và lọc hình thái học đa tầng (Morphological Opening & Closing).
"""

import cv2
import numpy as np


class RoadSegmenter:
    """
    Phân đoạn mặt đường và đo tỷ lệ chiếm dụng lòng đường thích nghi.
    Hỗ trợ cả chế độ MOG2 truyền thống và chế độ Hybrid Fusion kết hợp mô hình nhận diện.
    """

    def __init__(
        self,
        method: str = "MOG2",
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        min_contour_area: int = 150,
    ):
        self.method = method.upper()
        self.detect_shadows = detect_shadows
        self.min_contour_area = min_contour_area

        if self.method == "KNN":
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=400.0,
                detectShadows=detect_shadows,
            )
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows,
            )

        # Kernel dùng cho toán tử hình thái học
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        self.roi_mask = None
        self.roi_pts = None
        self.roi_pixel_count = 0
        self.detection_history = []

    def set_roi_polygon(self, frame_shape: tuple, polygon_pts: list):
        """
        Thiết lập vùng mặt đường quan tâm (ROI) bằng danh sách toạ độ đa giác.
        :param frame_shape: Kích thước khung hình (height, width)
        :param polygon_pts: Danh sách toạ độ [[x1, y1], [x2, y2], ...]
        """
        h, w = frame_shape[:2]
        self.roi_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(polygon_pts, dtype=np.int32)
        cv2.fillPoly(self.roi_mask, [pts], 255)
        self.roi_pts = pts
        self.roi_pixel_count = int(np.count_nonzero(self.roi_mask))

    def set_roi_mask(self, mask: np.ndarray):
        """
        Nhận trực tiếp mặt nạ nhị phân ROI.
        :param mask: numpy array 2D có cùng kích thước với frame
        """
        self.roi_mask = (mask > 0).astype(np.uint8) * 255
        self.roi_pixel_count = int(np.count_nonzero(self.roi_mask))

    def auto_estimate_road_roi(self, frame_shape: tuple, detections: list = None) -> np.ndarray:
        """
        Tự động ước lượng đa giác lòng đường thích nghi cho bất kỳ video nào dựa trên
        phân bố các phương tiện đang lưu thông trên tuyến đường.
        """
        h, w = frame_shape[:2]

        if detections:
            for d in detections:
                box = d.get("box", None)
                if box:
                    self.detection_history.append(box)

        # Giữ lại tối đa 150 hộp bao gần nhất
        if len(self.detection_history) > 150:
            self.detection_history = self.detection_history[-150:]

        # Nếu đã tích luỹ đủ vết tích phương tiện (ít nhất 6 xe), tính toán đường bao lồi
        if len(self.detection_history) >= 6:
            points = []
            for b in self.detection_history:
                bx1, by1, bx2, by2 = b
                points.append([bx1, by2])  # Điểm tiếp xúc mặt đường bên trái
                points.append([bx2, by2])  # Điểm tiếp xúc mặt đường bên phải
                points.append([int((bx1 + bx2) / 2), by1])

            pts_arr = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(pts_arr)

            # Tạo mặt nạ đa giác từ đường bao lồi và mở rộng nhẹ
            auto_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(auto_mask, hull, 255)

            # Đảm bảo phần đáy mặt đường phủ xuống gần cuối khung hình
            y_max = np.max(pts_arr[:, 1])
            if y_max < h * 0.90:
                cv2.rectangle(auto_mask, (0, int(h * 0.70)), (w, h), 255, -1)

            self.set_roi_mask(auto_mask)
            return auto_mask

        # Fallback chuẩn hình thang phối cảnh tự nhiên theo tỷ lệ khung hình
        default_trapezoid = np.float32([
            [int(w * 0.32), int(h * 0.28)],
            [int(w * 0.68), int(h * 0.28)],
            [int(w * 0.98), int(h * 0.98)],
            [int(w * 0.02), int(h * 0.98)],
        ])
        self.set_roi_polygon((h, w), default_trapezoid)
        return self.roi_mask

    def extract_occupancy(
        self,
        frame: np.ndarray,
        roi_mask: np.ndarray = None,
        learning_rate: float = -1,
        detections: list = None,
    ) -> tuple:
        """
        Phương thức tính toán tỷ lệ chiếm dụng lòng đường đạt độ chính xác cao.
        Tự động kích hoạt Hybrid Fusion (MOG2 + YOLO) nếu có danh sách detections.

        :param frame: Khung hình BGR hoặc Grayscale từ video
        :param roi_mask: Mặt nạ nhị phân vùng mặt đường (tuỳ chọn)
        :param learning_rate: Tốc độ học nền MOG2 (-1: mặc định)
        :param detections: Danh sách bounding boxes từ YOLOv8 [{'box': [x1, y1, x2, y2], ...}]
        :return: (occupancy_ratio, final_mask)
        """
        h, w = frame.shape[:2]

        # 1. Xác định vùng ROI sử dụng
        active_roi = None
        if roi_mask is not None:
            active_roi = (roi_mask > 0).astype(np.uint8) * 255
            total_roi_pixels = int(np.count_nonzero(active_roi))
        elif self.roi_mask is not None:
            active_roi = self.roi_mask
            total_roi_pixels = self.roi_pixel_count
        else:
            # Tự động ước lượng ROI thích nghi cho video mới
            active_roi = self.auto_estimate_road_roi((h, w), detections)
            total_roi_pixels = self.roi_pixel_count if self.roi_pixel_count > 0 else (h * w)

        # 2. Áp dụng thuật toán trừ nền MOG2
        raw_fg = self.bg_subtractor.apply(frame, learningRate=learning_rate)

        # 3. Khử bóng đổ (Shadow Suppression)
        if self.detect_shadows:
            _, fg_no_shadow = cv2.threshold(raw_fg, 250, 255, cv2.THRESH_BINARY)
        else:
            fg_no_shadow = raw_fg

        # 4. Xử lý hình thái học lọc nhiễu MOG2
        fg_cleaned = cv2.morphologyEx(fg_no_shadow, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        fg_cleaned = cv2.morphologyEx(fg_cleaned, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)

        # 5. ĐỘT PHÁ HYBRID FUSION: Hợp nhất mặt nạ MOG2 với diện tích hộp bao xe của YOLO
        # Đảm bảo khi xe dừng đỗ kẹt cứng không bao giờ bị đồng hóa vào nền
        if detections and len(detections) > 0:
            yolo_mask = np.zeros((h, w), dtype=np.uint8)
            for det in detections:
                box = det.get("box", None)
                if box:
                    bx1, by1, bx2, by2 = [int(v) for v in box]
                    bx1 = max(0, min(w - 1, bx1))
                    bx2 = max(0, min(w - 1, bx2))
                    by1 = max(0, min(h - 1, by1))
                    by2 = max(0, min(h - 1, by2))
                    if bx2 > bx1 and by2 > by1:
                        # Điền đặc thân xe
                        cv2.rectangle(yolo_mask, (bx1, by1), (bx2, by2), 255, -1)

            # Hợp nhất chuyển động MOG2 và hình chiếu xe YOLO
            combined_fg = cv2.bitwise_or(fg_cleaned, yolo_mask)
        else:
            combined_fg = fg_cleaned

        # 6. Giới hạn nghiêm ngặt trong vùng ROI mặt đường
        if active_roi is not None:
            final_mask = cv2.bitwise_and(combined_fg, active_roi)
        else:
            final_mask = combined_fg

        # 7. Tính tỷ lệ chiếm dụng mặt đường thực tế (Occupancy Ratio)
        vehicle_pixel_count = int(np.count_nonzero(final_mask))
        occupancy_ratio = 0.0
        if total_roi_pixels > 0:
            occupancy_ratio = float(np.clip(vehicle_pixel_count / total_roi_pixels, 0.0, 1.0))

        return occupancy_ratio, final_mask

    # Alias tương thích
    process_frame = extract_occupancy

    def visualize(self, frame: np.ndarray, fg_mask: np.ndarray, occupancy_ratio: float) -> np.ndarray:
        """
        Vẽ lớp phủ trực quan hóa phương tiện và chỉ số lên khung hình.
        """
        vis = frame.copy()

        # Vẽ viền vùng ROI (màu xanh cyan)
        if self.roi_mask is not None:
            roi_contours, _ = cv2.findContours(self.roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, roi_contours, -1, (255, 255, 0), 2)

        # Lớp phủ bán trong suốt màu đỏ cam lên phương tiện
        overlay = vis.copy()
        overlay[fg_mask > 0] = [0, 69, 255]
        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

        # Hiển thị thông số lên ảnh
        cv2.rectangle(vis, (15, 15), (420, 65), (0, 0, 0), -1)
        text = f"Occupancy Ratio: {occupancy_ratio * 100:.1f}%"
        color = (0, 255, 0) if occupancy_ratio < 0.3 else ((0, 255, 255) if occupancy_ratio < 0.7 else (0, 0, 255))
        cv2.putText(vis, text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        return vis


RoadOccupancySegmentor = RoadSegmenter
