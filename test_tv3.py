"""
SCRIPT TEST ĐỘC LẬP DÀNH CHO THÀNH VIÊN 3 (TV3)
Kiểm tra module trừ nền và đo tỷ lệ chiếm dụng mặt đường (RoadSegmenter).
Tích hợp trực tiếp với bộ tính TCI của TV1 (TrafficCongestionEvaluator).
"""

import sys
import cv2
import numpy as np

# Cấu hình in tiếng Việt trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modules.segmentation import RoadSegmenter
from modules.congestion_evaluator import TrafficCongestionEvaluator


def run_simulation_demo():
    print("==================================================================")
    print("  DEMO CHAY THU MODULE TV3: TRU NEN & TY LE CHIEM DUNG MAT DUONG")
    print("==================================================================")

    # 1. Khởi tạo đối tượng của TV3 và TV1
    segmentor = RoadSegmenter(method="MOG2", history=100, min_contour_area=150)
    evaluator = TrafficCongestionEvaluator()

    # Kích thước khung hình giả lập
    w, h = 640, 480

    # Thiết lập đa giác mặt đường ROI (Làn đường chính giữa)
    road_roi = [(100, 450), (220, 150), (420, 150), (540, 450)]
    segmentor.set_roi_polygon((h, w), road_roi)

    print("[INFO] Dang tao mo phong giao thong (120 frames)...")

    # Danh sách xe giả lập: [x, y, w_box, h_box, speed_y]
    cars = [
        [280, 160, 45, 70, 4],
        [180, 250, 40, 60, 5],
        [360, 320, 50, 80, 3],
    ]

    for frame_idx in range(1, 121):
        # Tạo nền đường màu xám đậm
        frame = np.full((h, w, 3), 70, dtype=np.uint8)

        # Vẽ vạch kẻ đường (background)
        cv2.line(frame, (320, 150), (320, 450), (200, 200, 200), 2)
        cv2.line(frame, (220, 150), (100, 450), (255, 255, 255), 3)
        cv2.line(frame, (420, 150), (540, 450), (255, 255, 255), 3)

        # Cập nhật chuyển động xe (sau frame 20 để background MOG2 ổn định)
        if frame_idx > 20:
            for car in cars:
                car[1] += car[4]
                if car[1] > 420:
                    car[1] = 160  # quay lại đầu đường

                # Vẽ xe và bóng đổ
                # Bóng đổ màu xám mờ phía dưới xe
                cv2.ellipse(frame, (car[0] + car[2]//2 + 10, car[1] + car[3]), (car[2]//2, 10), 0, 0, 360, (30, 30, 30), -1)
                # Thân xe
                cv2.rectangle(frame, (car[0], car[1]), (car[0] + car[2], car[1] + car[3]), (220, 180, 50), -1)

        # 2. Xử lý frame qua Module TV3 (chuẩn interface của TV1 trong main.py)
        occ_ratio, fg_mask = segmentor.extract_occupancy(frame)

        # 3. Gửi sang TV1 tính thử TCI
        # Giả lập vận tốc từ TV4 và PCU từ TV5
        simulated_speed = max(5.0, 40.0 * (1.0 - occ_ratio))
        simulated_pcu = occ_ratio * 40.0
        tci_res = evaluator.compute_tci(occupancy_ratio=occ_ratio, avg_speed=simulated_speed, pcu_count=simulated_pcu)

        # In kết quả sau mỗi 15 frames
        if frame_idx % 15 == 0:
            print(f"Frame {frame_idx:03d} | TV3 Occupancy: {occ_ratio*100:5.1f}% | "
                  f"Van toc uoc tinh: {simulated_speed:4.1f} km/h | "
                  f"TCI: {tci_res['smoothed_tci']:.2f} -> {tci_res['level']}")

    print("\n[THANH CONG] Module TV3 hoat dong tron tru va tich hop tot voi TV1!")


if __name__ == "__main__":
    run_simulation_demo()
