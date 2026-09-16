"""
SCRIPT KIỂM THỬ MODULE TV3: TRỪ NỀN & TỶ LỆ CHIẾM DỤNG MẶT ĐƯỜNG
Phụ trách: TV3 (Đặng Minh Quân)
Học phần: Xử lý ảnh và Thị giác máy tính (121036) - UTH
"""

import os
import sys
import time
import cv2
import numpy as np

# Cấu hình in tiếng Việt trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modules.segmentation import RoadSegmenter
from modules.congestion_evaluator import TrafficCongestionEvaluator

# Nạp cấu hình toạ độ từ configs nếu có
try:
    from configs.toadovideo import VIDEO_CONFIG
except ImportError:
    VIDEO_CONFIG = {}


def test_real_video(video_filename="traffic_congested.mp4", max_frames=120, show_window=False):
    """Kiểm thử trên video giao thông thực tế của nhóm."""
    video_path = os.path.join("data", "raw", video_filename)
    if not os.path.exists(video_path):
        print(f"[CANH BAO] Khong tim thay video {video_path}. Chuyen sang che do mo phong.")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[LOI] Khong the doc video: {video_path}")
        return False

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n==================================================================")
    print(f"  KIEM THU TV3 TREN VIDEO THUC TE: {video_filename}")
    print(f"  Do phan giai: {width}x{height} | FPS goc: {video_fps:.1f} | Tong frames: {total_video_frames}")
    print(f"==================================================================")

    # Khởi tạo RoadSegmenter và TrafficCongestionEvaluator
    segmentor = RoadSegmenter(method="MOG2", history=500, var_threshold=16.0, detect_shadows=True, min_contour_area=200)
    evaluator = TrafficCongestionEvaluator()

    # Cài đặt ROI từ toạ độ nhóm đã đo đạc
    if video_filename in VIDEO_CONFIG:
        pts = VIDEO_CONFIG[video_filename]["src_pts"]
        segmentor.set_roi_polygon((height, width), pts)
        print(f"[OK] Da cai dat toa do ROI mat duong tu configs/toadovideo.py ({len(pts)} diem).")
    else:
        print("[INFO] Khong co toa do rieng, su dung toan bo mat duong mac dinh.")

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)

    frame_count = 0
    total_time = 0.0
    occupancy_list = []
    sample_snapshot = None

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_t = time.perf_counter()

        # 1. TV3: Trừ nền và đo diện tích chiếm dụng
        occ_ratio, fg_mask = segmentor.extract_occupancy(frame)

        process_time = time.perf_counter() - start_t
        total_time += process_time
        occupancy_list.append(occ_ratio)

        # 2. TV1: Đánh giá TCI
        # Giả lập vận tốc tỉ lệ nghịch với occupancy và PCU tương quan
        est_speed = max(3.0, 40.0 * (1.0 - occ_ratio))
        est_pcu = occ_ratio * 45.0
        tci_res = evaluator.compute_tci(occupancy_ratio=occ_ratio, avg_speed=est_speed, pcu_count=est_pcu)

        # Trực quan hóa
        vis_frame = segmentor.visualize(frame, fg_mask, occ_ratio)

        # Lưu 1 frame snapshot tại frame 80 (khi mô hình nền đã hội tụ)
        if frame_count == min(80, max_frames):
            # Tạo ảnh ghép 3 khung hình: Gốc + Mask nhị phân + Lớp phủ trực quan
            h_small, w_small = 360, 640
            f_orig_small = cv2.resize(frame, (w_small, h_small))
            f_mask_small = cv2.resize(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR), (w_small, h_small))
            f_vis_small = cv2.resize(vis_frame, (w_small, h_small))

            cv2.putText(f_orig_small, "1. Anh goc", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(f_mask_small, "2. Mask phuong tien (khu bong)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(f_vis_small, f"3. Overlay | Occ: {occ_ratio*100:.1f}%", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            sample_snapshot = np.hstack([f_orig_small, f_mask_small, f_vis_small])
            save_path = os.path.join("data", "processed", "tv3_test_result.jpg")
            cv2.imwrite(save_path, sample_snapshot)
            print(f"[LUU ANH] Da luu anh chup so sanh vao: {save_path}")

        if frame_count % 20 == 0 or frame_count == max_frames:
            current_fps = frame_count / total_time if total_time > 0 else 0
            print(f"Frame {frame_count:03d}/{max_frames:03d} | "
                  f"TV3 Occupancy: {occ_ratio*100:5.1f}% | "
                  f"TCI: {tci_res['smoothed_tci']:.2f} ({tci_res['level']}) | "
                  f"Toc do: {current_fps:4.1f} FPS")

        if show_window:
            cv2.imshow("TV3 - Visualized Occupancy", vis_frame)
            cv2.imshow("TV3 - Foreground Mask", fg_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_occ = np.mean(occupancy_list) if occupancy_list else 0.0

    print("\n------------------------------------------------------------------")
    print(f"[KET QUA TONG KET - TV3]")
    print(f"  + So frame da xu ly : {frame_count} frames")
    print(f"  + Toc do trung binh : {avg_fps:.1f} FPS (Tieu chi de tai >= 20 FPS: {'DAT CHUAN' if avg_fps >= 20 else 'CHUA DAT'})")
    print(f"  + Ty le chiem dung TB: {avg_occ*100:.1f}%")
    print("------------------------------------------------------------------\n")
    return True


def run_simulation_demo():
    print("==================================================================")
    print("  DEMO MO PHONG GIAO THONG (SYNTHETIC SIMULATION)")
    print("==================================================================")
    segmentor = RoadSegmenter(method="MOG2", history=100, min_contour_area=150)
    evaluator = TrafficCongestionEvaluator()
    w, h = 640, 480
    road_roi = [(100, 450), (220, 150), (420, 150), (540, 450)]
    segmentor.set_roi_polygon((h, w), road_roi)

    cars = [
        [280, 160, 45, 70, 4],
        [180, 250, 40, 60, 5],
        [360, 320, 50, 80, 3],
    ]

    for frame_idx in range(1, 101):
        frame = np.full((h, w, 3), 70, dtype=np.uint8)
        cv2.line(frame, (320, 150), (320, 450), (200, 200, 200), 2)
        cv2.line(frame, (220, 150), (100, 450), (255, 255, 255), 3)
        cv2.line(frame, (420, 150), (540, 450), (255, 255, 255), 3)

        if frame_idx > 20:
            for car in cars:
                car[1] += car[4]
                if car[1] > 420:
                    car[1] = 160
                cv2.ellipse(frame, (car[0] + car[2]//2 + 10, car[1] + car[3]), (car[2]//2, 10), 0, 0, 360, (30, 30, 30), -1)
                cv2.rectangle(frame, (car[0], car[1]), (car[0] + car[2], car[1] + car[3]), (220, 180, 50), -1)

        occ_ratio, fg_mask = segmentor.extract_occupancy(frame)
        if frame_idx % 20 == 0:
            print(f"Frame {frame_idx:03d} | TV3 Occupancy: {occ_ratio*100:5.1f}%")

    print("[THANH CONG] Mo phong hoat dong hoan hao!")


if __name__ == "__main__":
    show_gui = "--show" in sys.argv
    video_target = "traffic_congested.mp4"
    for arg in sys.argv:
        if arg.endswith(".mp4"):
            video_target = arg

    success = test_real_video(video_target, max_frames=100, show_window=show_gui)
    if not success:
        run_simulation_demo()
