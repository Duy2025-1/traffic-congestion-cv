"""
MODULE PHÂN ĐOẠN & TỶ LỆ CHIẾM DỤNG MẶT ĐƯỜNG (ROAD OCCUPANCY)
Phụ trách: TV3
Học phần: Xử lý ảnh và Thị giác máy tính (121036) - UTH
"""

import cv2
import numpy as np


class RoadSegmenter:
    """
    Phân đoạn mặt đường và phát hiện diện tích chiếm dụng (Chương 4)
    Tương thích hoàn toàn với main pipeline của TV1 và hỗ trợ cấu hình nâng cao.
    """
    def __init__(
        self,
        method: str = "MOG2",
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        min_contour_area: int = 150
    ):
        """
        Khởi tạo bộ trừ nền và đo tỷ lệ chiếm dụng mặt đường.

        :param method: Thuật toán trừ nền ('MOG2' hoặc 'KNN')
        :param history: Số frame lịch sử dùng để học mô hình nền
        :param var_threshold: Ngưỡng phương sai (Mahalanobis distance) xác định foreground
        :param detect_shadows: Phát hiện bóng đổ (bóng = 127, vật thể = 255)
        :param min_contour_area: Diện tích đường bao tối thiểu để lọc nhiễu nhỏ
        """
        self.method = method.upper()
        self.detect_shadows = detect_shadows
        self.min_contour_area = min_contour_area

        # Khởi tạo thuật toán trừ nền
        if self.method == "KNN":
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=400.0,
                detectShadows=detect_shadows
            )
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows
            )

        # Kernel dùng cho toán tử hình thái học (Morphological Kernels)
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        self.roi_mask = None
        self.roi_pixel_count = 0

    def set_roi_polygon(self, frame_shape: tuple, polygon_pts: list):
        """
        Thiết lập vùng mặt đường quan tâm (ROI) bằng danh sách toạ độ đa giác.
        :param frame_shape: Kích thước khung hình (height, width)
        :param polygon_pts: Danh sách toạ độ [(x1, y1), (x2, y2), ...]
        """
        h, w = frame_shape[:2]
        self.roi_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(polygon_pts, dtype=np.int32)
        cv2.fillPoly(self.roi_mask, [pts], 255)
        self.roi_pixel_count = int(np.count_nonzero(self.roi_mask))

    def set_roi_mask(self, mask: np.ndarray):
        """
        Nhận trực tiếp mặt nạ nhị phân ROI (ví dụ nhận từ TV2).
        :param mask: numpy array 2D có cùng kích thước với frame
        """
        self.roi_mask = (mask > 0).astype(np.uint8) * 255
        self.roi_pixel_count = int(np.count_nonzero(self.roi_mask))

    def extract_occupancy(self, frame: np.ndarray, roi_mask: np.ndarray = None, learning_rate: float = -1) -> tuple:
        """
        Phương thức chính chuẩn interface cho main.py của TV1:
        Trừ nền, khử bóng, lọc hình thái học, tính occupancy ratio.

        :param frame: Khung hình BGR từ camera
        :param roi_mask: Mặt nạ nhị phân vùng mặt đường (tuỳ chọn)
        :param learning_rate: Tốc độ cập nhật nền (-1: mặc định, 0: đóng băng nền khi kẹt xe)
        :return: (occupancy_ratio, cleaned_mask)
        """
        h, w = frame.shape[:2]

        # 1. Xác định ROI mask sử dụng
        active_roi = None
        if roi_mask is not None:
            active_roi = (roi_mask > 0).astype(np.uint8) * 255
            total_roi_pixels = int(np.count_nonzero(active_roi))
        elif self.roi_mask is not None:
            active_roi = self.roi_mask
            total_roi_pixels = self.roi_pixel_count
        else:
            total_roi_pixels = h * w

        # 2. Áp dụng thuật toán trừ nền
        raw_fg = self.bg_subtractor.apply(frame, learningRate=learning_rate)

        # 3. Khử bóng đổ (Shadow Suppression)
        if self.detect_shadows:
            # MOG2 gán bóng đổ là 127, chỉ giữ lại pixel foreground thực sự (ngưỡng > 250)
            _, fg_no_shadow = cv2.threshold(raw_fg, 250, 255, cv2.THRESH_BINARY)
        else:
            fg_no_shadow = raw_fg

        # 4. Giới hạn trong vùng ROI mặt đường
        if active_roi is not None:
            fg_in_roi = cv2.bitwise_and(fg_no_shadow, active_roi)
        else:
            fg_in_roi = fg_no_shadow

        # 5. Xử lý hình thái học:
        # - Opening: loại bỏ nhiễu trắng li ti (lá cây, rung camera)
        # - Closing: hàn gắn thân xe bị rỗng ruột
        fg_cleaned = cv2.morphologyEx(fg_in_roi, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        fg_cleaned = cv2.morphologyEx(fg_cleaned, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)

        # 6. Lọc nhiễu theo diện tích contour
        contours, _ = cv2.findContours(fg_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(fg_cleaned)
        vehicle_pixel_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_contour_area:
                cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                vehicle_pixel_count += area

        # 7. Tính tỷ lệ chiếm dụng mặt đường (Occupancy Ratio)
        occupancy_ratio = 0.0
        if total_roi_pixels > 0:
            occupancy_ratio = float(np.clip(vehicle_pixel_count / total_roi_pixels, 0.0, 1.0))

        return occupancy_ratio, final_mask

    # Alias để tương thích cả 2 tên gọi
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
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

        # Hiển thị thông số lên ảnh
        cv2.rectangle(vis, (15, 15), (420, 65), (0, 0, 0), -1)
        text = f"Occupancy Ratio: {occupancy_ratio * 100:.1f}%"
        color = (0, 255, 0) if occupancy_ratio < 0.3 else ((0, 255, 255) if occupancy_ratio < 0.7 else (0, 0, 255))
        cv2.putText(vis, text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        return vis


# Alias tương thích
RoadOccupancySegmentor = RoadSegmenter
