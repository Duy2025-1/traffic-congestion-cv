"""
SCRIPT KIỂM THỬ MODULE TV5: NHẬN DẠNG PHƯƠNG TIỆN & QUY ĐỔI TẢI TRỌNG PCU
Phụ trách: TV5
Học phần: Xử lý ảnh và Thị giác máy tính (121036) - UTH
Dự án: Hệ thống Giám sát và Đánh giá Ùn tắc Giao thông Đô thị

Chức năng:
1. Nạp và xử lý video giao thông thực tế (traffic_congested, traffic_free_flow, traffic_traffic_light).
2. Tách và đếm 4 nhóm phương tiện: Xe máy (0.33 PCU), Ô tô con (1.0 PCU), Xe buýt (2.5 PCU), Xe tải (3.0 PCU).
3. Khảo sát tham số (Parameter Sweep) trên các ngưỡng conf_threshold và iou_threshold.
4. Tự động lưu hình ảnh bounding box trực quan và biểu đồ khảo sát vào data/processed/.
"""

import os
import sys
import time
import cv2
import numpy as np

# Cấu hình encoding console UTF-8 trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm thư mục gốc vào đường dẫn hệ thống
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.detection import VehicleDetector
from config import PCU_WEIGHTS


def run_video_detection(video_path, max_frames=150, show_window=False, save_result_path="data/processed/tv5_test_result.jpg"):
    """
    Chạy nhận dạng phương tiện và tính tải trọng PCU trên video thực tế.
    """
    if not os.path.exists(video_path):
        print(f"[CANH BAO] Khong tim thay file video: {video_path}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[LOI] Khong the mo video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n" + "=" * 70)
    print(f"  KIEM THU TV5 TREN VIDEO: {os.path.basename(video_path)}")
    print(f"  Do phan giai: {w}x{h} | FPS goc: {fps_video:.1f} | Tong frames: {total_frames}")
    print("=" * 70)

    detector = VehicleDetector(conf_thresh=0.4, iou_thresh=0.45)
    os.makedirs(os.path.dirname(save_result_path), exist_ok=True)

    frame_idx = 0
    pcu_history = []
    vehicle_history = []
    fps_history = []
    saved_sample_frame = None

    start_total_time = time.time()

    while cap.isOpened() and frame_idx < max_frames:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Nhận dạng và tính PCU
        total_pcu, counts, detections = detector.detect_and_count_pcu(frame, return_counts=True)
        pcu_history.append(total_pcu)
        vehicle_history.append(counts)

        t1 = time.time()
        fps_curr = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
        fps_history.append(fps_curr)

        # Vẽ trực quan hóa
        annotated_frame = detector.draw_detections(frame, detections, draw_hud=True)

        # Lưu lại frame mẫu có nhiều xe nhất
        if saved_sample_frame is None or sum(counts.values()) > saved_sample_frame[1]:
            saved_sample_frame = (annotated_frame, sum(counts.values()), total_pcu, frame_idx)

        # In thông số mỗi 30 frames
        if frame_idx % 30 == 0 or frame_idx == 1:
            print(f"[Frame {frame_idx:03d}/{max_frames}] "
                  f"Xe may: {counts['motorcycle']:2d} | Oto: {counts['car']:2d} | "
                  f"Buyt: {counts['bus']:2d} | Tai: {counts['truck']:2d} || "
                  f"Tong PCU: {total_pcu:5.2f} | FPS: {fps_curr:4.1f}")

        if show_window:
            cv2.imshow("TV5 - Vehicle Detection & PCU Estimation", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    total_elapsed = time.time() - start_total_time
    avg_fps = np.mean(fps_history) if fps_history else 0
    avg_pcu = np.mean(pcu_history) if pcu_history else 0
    max_pcu = np.max(pcu_history) if pcu_history else 0

    print("-" * 70)
    print(f"[KET QUA] Da xu ly {frame_idx} frames trong {total_elapsed:.2f}s (Trung binh: {avg_fps:.1f} FPS)")
    print(f"[THONG KE PCU] Trung binh: {avg_pcu:.2f} | Cao nhat: {max_pcu:.2f}")

    if saved_sample_frame is not None:
        cv2.imwrite(save_result_path, saved_sample_frame[0])
        print(f"[OK] Da luu frame truc quan tai frame {saved_sample_frame[3]} vao: {save_result_path}")

    return {
        "frames_processed": frame_idx,
        "avg_fps": avg_fps,
        "avg_pcu": avg_pcu,
        "max_pcu": max_pcu,
        "history_pcu": pcu_history
    }


def run_parameter_sweep_experiment(video_path, target_msec=3000):
    """
    Thực hiện khảo sát Parameter Sweep (ngưỡng tin cậy conf_threshold và iou_threshold)
    trên 1 frame tiêu biểu để phân tích sự tách biệt giữa xe buýt và xe tải.
    """
    print("\n" + "=" * 70)
    print("  KHAO SAT THAM SO (PARAMETER SWEEP) TV5: XE BUYT vs XE TAI")
    print("=" * 70)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[LOI] Khong the mo video de khao sat: {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[LOI] Khong doc duoc frame khao sat.")
        return

    detector = VehicleDetector()
    conf_list = [0.25, 0.40, 0.55, 0.70]
    iou_list = [0.30, 0.45, 0.60]

    sweep_data = detector.sweep_parameters(frame, conf_thresholds=conf_list, iou_thresholds=iou_list)

    print(f"\n{'Conf':>6} | {'IoU':>5} | {'Xe May':>7} | {'O To':>6} | {'Xe Buyt':>8} | {'Xe Tai':>7} | {'Tong Xe':>8} | {'Tong PCU':>9}")
    print("-" * 75)
    for r in sweep_data:
        print(f"{r['conf_threshold']:6.2f} | {r['iou_threshold']:5.2f} | "
              f"{r['motorcycle']:7d} | {r['car']:6d} | {r['bus']:8d} | {r['truck']:7d} | "
              f"{r['total_vehicles']:8d} | {r['total_pcu']:9.2f}")

    # Vẽ và lưu biểu đồ trực quan hóa kết quả khảo sát
    try:
        import matplotlib.pyplot as plt
        os.makedirs("data/processed/detection_sweep", exist_ok=True)

        confs = sorted(list(set(r["conf_threshold"] for r in sweep_data)))
        bus_by_conf = [next(r["bus"] for r in sweep_data if r["conf_threshold"] == c and r["iou_threshold"] == 0.45) for c in confs]
        truck_by_conf = [next(r["truck"] for r in sweep_data if r["conf_threshold"] == c and r["iou_threshold"] == 0.45) for c in confs]
        pcu_by_conf = [next(r["total_pcu"] for r in sweep_data if r["conf_threshold"] == c and r["iou_threshold"] == 0.45) for c in confs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Đồ thị 1: Số lượng xe buýt vs xe tải theo ngưỡng Confidence
        ax1.plot(confs, bus_by_conf, marker='o', color='orange', linewidth=2, label='Xe buýt (bus)')
        ax1.plot(confs, truck_by_conf, marker='s', color='red', linewidth=2, label='Xe tải (truck)')
        ax1.set_title("Số lượng nhận dạng Xe Buýt vs Xe Tải theo Confidence Thresh (IoU=0.45)")
        ax1.set_xlabel("Confidence Threshold")
        ax1.set_ylabel("Số lượng phát hiện")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        # Đồ thị 2: Tổng tải trọng PCU
        ax2.plot(confs, pcu_by_conf, marker='^', color='purple', linewidth=2, label='Tổng PCU')
        ax2.set_title("Tổng Điểm Tải Trọng PCU theo Confidence Threshold")
        ax2.set_xlabel("Confidence Threshold")
        ax2.set_ylabel("PCU")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        chart_path = "data/processed/detection_sweep/tv5_parameter_sweep.png"
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"[OK] Da luu bieu do khao sat tham so vao: {chart_path}")

    except Exception as e:
        print(f"[CANH BAO] Khong the xuat bieu do matplotlib: {e}")


def main():
    print("=========================================================")
    print("  CHUONG TRINH KIEM THU MODULE TV5: DETECTION & PCU")
    print("  Nhom 5 - Mon: Xu ly anh va Thi giac may tinh (UTH)")
    print("=========================================================")

    videos = [
        "traffic_congested.mp4",
        "traffic_free_flow.mp4",
        "traffic_traffic_light.mp4"
    ]
    congested_path = os.path.join("data", "raw", "traffic_congested.mp4")

    # Xử lý các cờ dòng lệnh
    if "--sweep" in sys.argv:
        run_parameter_sweep_experiment(congested_path)
        return
    elif "--all" in sys.argv:
        run_video_detection(congested_path, max_frames=100)
        run_parameter_sweep_experiment(congested_path)
        return
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        selected_video = sys.argv[1]
        run_video_detection(selected_video, max_frames=120)
        return

    # Nếu không phải terminal tương tác (ví dụ chạy qua pipe/script)
    if not sys.stdin.isatty():
        print("[INFO] Che do non-interactive: Tu dong chay toan bo quy trinh kiem thu.")
        run_video_detection(congested_path, max_frames=60)
        run_parameter_sweep_experiment(congested_path)
        return

    print("\nDanh sach video co san trong data/raw/:")
    for idx, v in enumerate(videos, start=1):
        v_path = os.path.join("data", "raw", v)
        status = "CO SAN" if os.path.exists(v_path) else "CHUA CO"
        print(f"  {idx}. {v} [{status}]")
    print("  4. Chay khao sat Parameter Sweep nhanh (Frame 3s traffic_congested)")
    print("  5. Chay toan bo (Kiem thu video + Parameter Sweep)")

    try:
        choice = input("\nChon che do kiem thu (1-5) [Mac dinh 5]: ").strip()
        if not choice:
            choice = "5"
    except (EOFError, KeyboardInterrupt):
        choice = "5"

    if choice in ["1", "2", "3"]:
        selected_video = os.path.join("data", "raw", videos[int(choice) - 1])
        run_video_detection(selected_video, max_frames=150)
    elif choice == "4":
        run_parameter_sweep_experiment(congested_path)
    elif choice == "5":
        run_video_detection(congested_path, max_frames=120)
        run_parameter_sweep_experiment(congested_path)


if __name__ == "__main__":
    main()
