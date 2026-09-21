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


# ============================================================
# PERSPECTIVE ROI ESTIMATOR
# Xác định ROI mặt đường dựa trên hình học phối cảnh thực tế.
# Nguyên lý:
#   1. Phát hiện vạch kẻ đường/mép đường bằng Canny + HoughLinesP
#   2. Tách đường thẳng trái (slope < 0) / phải (slope > 0) theo góc nghiêng
#   3. Tìm điểm hội tụ phối cảnh (Vanishing Point) = giao điểm 2 nhóm
#   4. Xây dựng polygon ROI hình thang phối cảnh từ VP xuống đáy frame
# ============================================================
class PerspectiveROIEstimator:
    """
    Tự động ước lượng ROI mặt đường dựa trên hình học phối cảnh.
    Không dùng tọa độ cố định tùy ý — ROI được suy diễn từ
    các đường thẳng phối cảnh phát hiện trong frame thực tế.
    """

    # Ngưỡng độ dốc hợp lệ: loại đường ngang (|slope| quá nhỏ) và gần thẳng đứng
    SLOPE_MIN = 0.2   # |tan(θ)| tối thiểu ~ 11°
    SLOPE_MAX = 10.0  # |tan(θ)| tối đa ~ 84°

    def __init__(self):
        self._vp_history = []      # Lịch sử vanishing point để làm mượt
        self._max_vp_history = 30

    # ------------------------------------------------------------------
    def estimate_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Ước lượng polygon ROI hình thang phối cảnh từ một frame BGR/gray.

        Quy trình:
          Gray → Canny edge → HoughLinesP → phân tách trái/phải →
          fit line mỗi nhóm → tìm vanishing point →
          kéo hai cạnh từ VP xuống đáy frame → thành polygon.

        :return: mảng numpy int32 shape (4,2) theo thứ tự:
                 [top-left, top-right, bottom-right, bottom-left]
                 Trả về trapezoid mặc định nếu không phát hiện đủ đường.
        """
        h, w = frame.shape[:2]

        # 1. Chuyển xám + cân bằng tương phản nhẹ
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Chỉ xét nửa dưới frame (phần mặt đường gần camera rõ ràng hơn)
        roi_y_start = int(h * 0.35)
        gray_roi = gray[roi_y_start:, :]

        # Làm mờ nhẹ để giảm nhiễu trước Canny
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3)

        # 2. Phát hiện đoạn thẳng bằng HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=60,
            maxLineGap=30,
        )

        if lines is None or len(lines) < 4:
            return self._default_trapezoid(h, w)

        # 3. Phân loại đường trái / phải theo độ dốc
        left_lines  = []  # slope < 0  (đường mép trái hội tụ lên VP từ dưới-trái)
        right_lines = []  # slope > 0  (đường mép phải hội tụ lên VP từ dưới-phải)

        for seg in lines:
            # HoughLinesP trả về shape (N,1,4) hoặc (N,4) tùy phiên bản OpenCV
            coords = seg.flatten()
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            if x2 == x1:
                continue  # bỏ đường thẳng đứng
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < self.SLOPE_MIN or abs(slope) > self.SLOPE_MAX:
                continue
            # Bù lại offset y do cắt ROI
            y1_full = y1 + roi_y_start
            y2_full = y2 + roi_y_start
            if slope < 0:
                left_lines.append((x1, y1_full, x2, y2_full, slope))
            else:
                right_lines.append((x1, y1_full, x2, y2_full, slope))

        if len(left_lines) < 2 or len(right_lines) < 2:
            return self._default_trapezoid(h, w)

        # 4. Fit đường đại diện cho nhóm trái & phải (trung bình có trọng số theo chiều dài)
        left_rep  = self._fit_representative_line(left_lines,  h)
        right_rep = self._fit_representative_line(right_lines, h)

        if left_rep is None or right_rep is None:
            return self._default_trapezoid(h, w)

        # 5. Tìm vanishing point = giao điểm hai đường đại diện
        vp = self._line_intersection(left_rep, right_rep)
        if vp is None:
            return self._default_trapezoid(h, w)

        vp_x, vp_y = vp

        # VP phải nằm trong nửa trên frame (phối cảnh thực tế)
        if not (0 < vp_y < h * 0.70):
            return self._default_trapezoid(h, w)

        # Làm mượt VP qua lịch sử
        self._vp_history.append((vp_x, vp_y))
        if len(self._vp_history) > self._max_vp_history:
            self._vp_history = self._vp_history[-self._max_vp_history:]
        vp_x = int(np.median([v[0] for v in self._vp_history]))
        vp_y = int(np.median([v[1] for v in self._vp_history]))

        # 6. Xây dựng polygon hình thang từ VP xuống đáy frame
        #    - Cạnh trên: một đoạn nhỏ gần VP (vùng đường xa)
        #    - Cạnh dưới: toàn bộ chiều rộng mặt đường gần camera
        top_y    = min(int(vp_y + h * 0.06), int(h * 0.50))  # cạnh trên, sát VP
        bottom_y = int(h * 0.97)                               # cạnh dưới, gần mép frame

        # Tính x trên đường trái/phải tại top_y và bottom_y
        tl_x = self._x_at_y(left_rep,  top_y)
        tr_x = self._x_at_y(right_rep, top_y)
        bl_x = self._x_at_y(left_rep,  bottom_y)
        br_x = self._x_at_y(right_rep, bottom_y)

        # Đảm bảo thứ tự trái < phải (tránh polygon bị tréo do slope nhầm nhóm)
        if tl_x > tr_x:
            tl_x, tr_x = tr_x, tl_x
        if bl_x > br_x:
            bl_x, br_x = br_x, bl_x

        # Kẹp vào biên frame, thêm padding nhẹ ở đáy để không cắt xe sát mép
        pad = int(w * 0.02)
        bl_x = max(0,     bl_x - pad)
        br_x = min(w - 1, br_x + pad)
        tl_x = max(0,     tl_x)
        tr_x = min(w - 1, tr_x)

        # Kiểm tra chiều rộng polygon tối thiểu (phải rộng hơn 5% frame ở cạnh trên)
        if tr_x - tl_x < w * 0.05:
            return self._default_trapezoid(h, w)

        polygon = np.array([
            [tl_x, top_y],
            [tr_x, top_y],
            [br_x, bottom_y],
            [bl_x, bottom_y],
        ], dtype=np.int32)

        return polygon


    # ------------------------------------------------------------------
    def _fit_representative_line(self, line_group: list, h: int):
        """
        Tính đường đại diện (slope, intercept) cho một nhóm đường thẳng
        bằng trung bình có trọng số theo chiều dài đoạn.
        Trả về (slope_mean, intercept_mean) hoặc None nếu thất bại.
        """
        slopes, intercepts, weights = [], [], []
        for x1, y1, x2, y2, slope in line_group:
            length = np.hypot(x2 - x1, y2 - y1)
            intercept = y1 - slope * x1
            slopes.append(slope)
            intercepts.append(intercept)
            weights.append(length)
        total_w = sum(weights)
        if total_w == 0:
            return None
        s = sum(sl * wt for sl, wt in zip(slopes, weights)) / total_w
        b = sum(ic * wt for ic, wt in zip(intercepts, weights)) / total_w
        return (s, b)  # y = s*x + b  →  x = (y - b) / s

    def _x_at_y(self, line_sb, y: int) -> int:
        """Tính x theo y từ đường (slope, intercept): x = (y - b) / slope"""
        s, b = line_sb
        if abs(s) < 1e-6:
            return 0
        return int((y - b) / s)

    def _line_intersection(self, line1, line2):
        """
        Tìm giao điểm hai đường y = s1*x + b1 và y = s2*x + b2.
        Trả về (x, y) int hoặc None nếu song song.
        """
        s1, b1 = line1
        s2, b2 = line2
        denom = s1 - s2
        if abs(denom) < 1e-6:
            return None
        x = (b2 - b1) / denom
        y = s1 * x + b1
        return (int(x), int(y))

    def _default_trapezoid(self, h: int, w: int) -> np.ndarray:
        """
        Hình thang phối cảnh mặc định khi không phát hiện đủ đường thẳng.
        Tỉ lệ được chọn phù hợp với camera giao thông góc nhìn từ cao:
          - Cạnh trên hẹp (38%–62% chiều rộng) tượng trưng điểm xa
          - Cạnh dưới rộng (3%–97% chiều rộng) phủ mặt đường gần camera
        """
        return np.array([
            [int(w * 0.38), int(h * 0.35)],
            [int(w * 0.62), int(h * 0.35)],
            [int(w * 0.97), int(h * 0.97)],
            [int(w * 0.03), int(h * 0.97)],
        ], dtype=np.int32)


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

        # Bộ ước lượng ROI dựa trên hình học phối cảnh (dùng khi video mới/không có config)
        self.perspective_estimator = PerspectiveROIEstimator()
        self._perspective_roi_ready = False  # Đã thiết lập ROI từ perspective chưa


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

    def auto_estimate_road_roi(
        self,
        frame_shape: tuple,
        detections: list = None,
        frame: np.ndarray = None,
    ) -> np.ndarray:
        """
        Tự động ước lượng ROI mặt đường dựa trên hình học phối cảnh.

        Chiến lược 2 tầng:
          Tầng 1 (Perspective Geometry): Nếu có `frame`, dùng PerspectiveROIEstimator
            phát hiện vanishing point từ vạch kẻ đường → xây dựng polygon hình thang
            phối cảnh thực sự. Kết quả không phụ thuộc vào tọa độ cố định.
          Tầng 2 (Detection History): Sau khi tích lũy ≥ 20 phát hiện xe,
            tinh chỉnh cạnh trái/phải của ROI theo vị trí thực tế các phương tiện
            để ROI bám sát mặt đường hơn.

        :param frame_shape: (height, width) của frame
        :param detections: Danh sách bounding box từ YOLO (tuỳ chọn)
        :param frame: Frame BGR gốc để chạy perspective estimation (tuỳ chọn)
        """
        h, w = frame_shape[:2]

        # --- Tích lũy vết tích phương tiện ---
        if detections:
            for d in detections:
                box = d.get("box", None)
                if box:
                    self.detection_history.append(box)
        if len(self.detection_history) > 150:
            self.detection_history = self.detection_history[-150:]

        # =========================================================
        # TẦNG 1: Perspective-based ROI từ hình học mặt đường
        # =========================================================
        if frame is not None and not self._perspective_roi_ready:
            perspective_polygon = self.perspective_estimator.estimate_from_frame(frame)
            h_p, w_p = frame.shape[:2]
            # Kiểm tra polygon có hợp lệ (không phải default hoàn toàn)
            poly_area = cv2.contourArea(perspective_polygon)
            frame_area = h_p * w_p
            if poly_area > frame_area * 0.10:  # ROI phải chiếm ít nhất 10% khung hình
                self.set_roi_polygon((h_p, w_p), perspective_polygon)
                self.roi_pts = perspective_polygon
                self._perspective_roi_ready = True
                print(f"[PerspectiveROI] VP detected → ROI area = {poly_area/frame_area*100:.1f}% of frame")
                return self.roi_mask

        # =========================================================
        # TẦNG 2: Tinh chỉnh ROI theo lịch sử phát hiện phương tiện
        # Khi đã tích lũy đủ xe, dùng convex hull để tinh chỉnh
        # cạnh trái/phải của ROI bám sát vị trí thực tế các xe
        # =========================================================
        if len(self.detection_history) >= 20:
            # Lấy các điểm tiếp xúc bánh xe với mặt đường (đáy bounding box)
            bottom_points = []
            top_center_points = []
            for bx1, by1, bx2, by2 in self.detection_history:
                pad_x = int((bx2 - bx1) * 0.05)
                bottom_points.append([max(0, bx1 - pad_x), by2])
                bottom_points.append([min(w - 1, bx2 + pad_x), by2])
                top_center_points.append([int((bx1 + bx2) / 2), by1])

            all_points = np.array(bottom_points + top_center_points, dtype=np.int32)
            hull = cv2.convexHull(all_points)
            hull_area = cv2.contourArea(hull)

            if hull_area > (h * w) * 0.08:  # Hull có diện tích hợp lý
                auto_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(auto_mask, hull, 255)

                # Bảo đảm phần đáy phủ đủ (xe sát camera)
                y_max = int(np.max(all_points[:, 1]))
                if y_max < h * 0.85:
                    # Mở rộng xuống đáy theo đúng chiều rộng xe đã phát hiện
                    x_coords = all_points[:, 0]
                    x_lo = max(0,     int(np.percentile(x_coords, 5))  - int(w * 0.03))
                    x_hi = min(w - 1, int(np.percentile(x_coords, 95)) + int(w * 0.03))
                    cv2.rectangle(auto_mask, (x_lo, y_max), (x_hi, h - 1), 255, -1)

                self.set_roi_mask(auto_mask)
                return auto_mask

        # =========================================================
        # TẦNG 3: Fallback hình thang phối cảnh mặc định
        # Chỉ dùng khi cả 2 tầng trên không thành công
        # =========================================================
        fallback = self.perspective_estimator._default_trapezoid(h, w)
        self.set_roi_polygon((h, w), fallback)
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
        Đường viền ROI hiển thị màu VÀNG (BGR: 0, 255, 255) theo yêu cầu.
        """
        vis = frame.copy()

        # Vẽ viền vùng ROI màu VÀNG để dễ kiểm tra trực quan
        # Ưu tiên vẽ trực tiếp từ polygon points (sắc nét hơn findContours)
        if self.roi_pts is not None:
            cv2.polylines(vis, [self.roi_pts.reshape(-1, 1, 2)], isClosed=True,
                          color=(0, 255, 255), thickness=2)
        elif self.roi_mask is not None:
            roi_contours, _ = cv2.findContours(self.roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, roi_contours, -1, (0, 255, 255), 2)

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
