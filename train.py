"""
HỆ THỐNG HUẤN LUYỆN & FINE-TUNING MÔ HÌNH NHẬN DẠNG PHƯƠNG TIỆN (YOLOV8)
Dự án: Giám sát và Đánh giá Ùn tắc Giao thông Đô thị - UTH
File: train.py

Các tính năng nổi bật:
1. Hỗ trợ đa nguồn video linh hoạt: Tự động trích xuất frame và gán nhãn từ nhiều video bất kỳ
   thông qua danh sách tệp (--videos) hoặc thư mục (--video-dir) với các định dạng .mp4, .avi, .mov, .mkv.
2. Cơ chế gán nhãn 2 tầng tăng độ chính xác (High-Accuracy Two-Pass Pseudo-Labeling):
   - Tầng 1: Quét toàn khung hình 1280p.
   - Tầng 2: Cắt lát tăng cường vùng xa chân trời (Horizon Slicing) để phát hiện xe nhỏ ở xa.
   - Hợp nhất bằng Per-Class NMS bảo toàn các cụm xe máy đi liền kề.
3. Siêu tham số tối ưu hóa cho mật độ giao thông đô thị Việt Nam:
   - Augmentation: Mosaic (1.0), Mixup (0.15) chống che khuất (occlusion).
   - Tối ưu hóa định vị: Trọng số Box (7.5) và DFL (1.5).
4. Tự động lưu và cập nhật trọng số tối ưu vào weights/best.pt để hệ thống nhận diện nạp ngay.
"""

import os
import sys
import shutil
import argparse
import time
import glob
import cv2
import numpy as np

# Cấu hình encoding UTF-8 trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ánh xạ từ COCO sang 4 lớp phương tiện giao thông chuẩn hóa PCU
COCO_TO_TRAFFIC = {
    1: 0,  # bicycle -> motorcycle (xe đạp / xe máy điện)
    3: 0,  # motorcycle -> motorcycle (0.33 PCU)
    2: 1,  # car -> car (1.00 PCU)
    5: 2,  # bus -> bus (2.50 PCU)
    7: 3,  # truck -> truck (3.00 PCU)
}

CLASS_NAMES = ["motorcycle", "car", "bus", "truck"]
SUPPORTED_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")


def collect_video_files(video_inputs=None, video_dir=None):
    """
    Thu thập danh sách tất cả các tệp video từ danh sách hoặc thư mục.
    """
    video_files = []

    # 1. Từ danh sách video cụ thể
    if video_inputs:
        for item in video_inputs:
            if os.path.isfile(item) and item.endswith(SUPPORTED_EXTS):
                video_files.append(os.path.abspath(item))
            elif os.path.isdir(item):
                for ext in SUPPORTED_EXTS:
                    video_files.extend(glob.glob(os.path.join(item, f"*{ext}")))
                    video_files.extend(glob.glob(os.path.join(item, f"**/*{ext}"), recursive=True))

    # 2. Từ thư mục video mặc định nếu không chỉ định tệp
    if not video_files and video_dir:
        if os.path.exists(video_dir):
            for ext in SUPPORTED_EXTS:
                video_files.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
                video_files.extend(glob.glob(os.path.join(video_dir, f"**/*{ext}"), recursive=True))

    video_files = sorted(list(set(video_files)))
    return video_files


def prepare_dataset_from_videos(
    video_files=None,
    output_dir="data/dataset",
    frame_step=25,
    max_frames_per_video=40,
    base_model="yolov8n.pt",
    conf_thresh=0.20,
    iou_thresh=0.45,
):
    """
    Trích xuất khung hình từ nhiều video bất kỳ và tự động gán nhãn 4 lớp chuẩn PCU
    bằng cơ chế Two-Pass Slicing Detection để đạt độ chính xác tối đa.
    """
    from ultralytics import YOLO

    print("\n" + "=" * 75)
    print("  CHUẨN BỊ TẬP DỮ LIỆU ĐA VIDEO (MULTI-VIDEO DATASET PREPARATION)")
    print("=" * 75)

    if not video_files:
        print("[LỖI] Không tìm thấy tệp video nào để xử lý.")
        return False

    print(f"[INFO] Tìm thấy {len(video_files)} video phục vụ trích xuất:")
    for idx, v in enumerate(video_files, 1):
        print(f"   {idx}. {os.path.basename(v)} ({v})")
    print("-" * 75)

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Sử dụng mô hình nền chuẩn độ chính xác cao để gán nhãn
    model_weight = base_model if base_model else "yolov8n.pt"
    print(f"[INFO] Đang nạp mô hình gán nhãn giả chất lượng cao: {model_weight} ...")
    model = YOLO(model_weight)

    total_extracted = 0
    total_annotations = 0

    for vid_idx, vid_path in enumerate(video_files, 1):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"[CẢNH BÁO] Không thể mở video: {vid_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        extracted_for_vid = 0

        print(f"\n[{vid_idx}/{len(video_files)}] Đang trích xuất video: {vid_name} (Tổng {total_frames} frames)...")

        while cap.isOpened() and extracted_for_vid < max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_step != 0:
                continue

            h, w = frame.shape[:2]

            # Phân chia 80% train, 20% validation
            is_val = (extracted_for_vid % 5 == 0)
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

            file_stem = f"{vid_name}_f{frame_idx:05d}"
            img_out = os.path.join(target_img_dir, f"{file_stem}.jpg")
            lbl_out = os.path.join(target_lbl_dir, f"{file_stem}.txt")

            cv2.imwrite(img_out, frame)

            # 1. Quét tầng 1: Toàn khung hình độ phân giải cao 1280p
            res_full = model(
                frame,
                classes=list(COCO_TO_TRAFFIC.keys()) if hasattr(model, "names") and len(model.names) > 10 else None,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=1280,
                verbose=False,
            )[0]

            raw_boxes = []
            raw_scores = []
            raw_labels = []

            for box in res_full.boxes:
                cls_id = int(box.cls[0].item())
                if hasattr(model, "names") and len(model.names) > 10:
                    traffic_cls = COCO_TO_TRAFFIC.get(cls_id, None)
                else:
                    traffic_cls = cls_id if cls_id in [0, 1, 2, 3] else None

                if traffic_cls is not None:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
                    raw_scores.append(float(box.conf[0].item()))
                    raw_labels.append(traffic_cls)

            # 2. Quét tầng 2: Cắt lát vùng xa chân trời (Horizon Slicing) bắt xe nhỏ ở xa
            y_start, y_end = int(0.20 * h), int(0.65 * h)
            crop_distant = frame[y_start:y_end, :]
            res_distant = model(
                crop_distant,
                classes=list(COCO_TO_TRAFFIC.keys()) if hasattr(model, "names") and len(model.names) > 10 else None,
                conf=max(0.18, conf_thresh - 0.03),
                iou=iou_thresh,
                imgsz=1280,
                verbose=False,
            )[0]

            for box in res_distant.boxes:
                cls_id = int(box.cls[0].item())
                if hasattr(model, "names") and len(model.names) > 10:
                    traffic_cls = COCO_TO_TRAFFIC.get(cls_id, None)
                else:
                    traffic_cls = cls_id if cls_id in [0, 1, 2, 3] else None

                if traffic_cls is not None:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    real_y1 = y1 + y_start
                    real_y2 = y2 + y_start
                    raw_boxes.append([x1, real_y1, x2 - x1, real_y2 - real_y1])
                    raw_scores.append(float(box.conf[0].item()))
                    raw_labels.append(traffic_cls)

            # 3. Hợp nhất Per-Class NMS (Không làm mất xe máy đi sát nhau)
            label_lines = []
            if raw_boxes:
                for target_cls in [0, 1, 2, 3]:
                    indices_cls = [i for i, l in enumerate(raw_labels) if l == target_cls]
                    if not indices_cls:
                        continue
                    sub_b = [raw_boxes[i] for i in indices_cls]
                    sub_s = [raw_scores[i] for i in indices_cls]
                    nms_idx = cv2.dnn.NMSBoxes(sub_b, sub_s, conf_thresh, iou_thresh)
                    if len(nms_idx) > 0:
                        for k in nms_idx.flatten():
                            orig_i = indices_cls[k]
                            bx, by, bw, bh = raw_boxes[orig_i]
                            # Chuẩn hóa toạ độ YOLO [center_x, center_y, width, height]
                            x_c = (bx + bw / 2.0) / w
                            y_c = (by + bh / 2.0) / h
                            norm_w = bw / w
                            norm_h = bh / h
                            x_c = np.clip(x_c, 0.0, 1.0)
                            y_c = np.clip(y_c, 0.0, 1.0)
                            norm_w = np.clip(norm_w, 0.0, 1.0)
                            norm_h = np.clip(norm_h, 0.0, 1.0)
                            label_lines.append(f"{target_cls} {x_c:.6f} {y_c:.6f} {norm_w:.6f} {norm_h:.6f}")
                            total_annotations += 1

            with open(lbl_out, "w", encoding="utf-8") as f_lbl:
                f_lbl.write("\n".join(label_lines) + "\n")

            extracted_for_vid += 1
            total_extracted += 1

        cap.release()
        print(f"   -> Đã trích xuất {extracted_for_vid} frames chất lượng cao từ {vid_name}")

    print("-" * 75)
    print(f"[HOÀN TẤT] Tổng cộng đã tạo {total_extracted} ảnh và {total_annotations} nhãn bounding box.")
    print(f"  - Tập Train: {len(os.listdir(train_img_dir))} ảnh | Tập Val: {len(os.listdir(val_img_dir))} ảnh")
    print(f"[OK] Dữ liệu sẵn sàng để huấn luyện tại: {output_dir}")
    print("=" * 75)
    return True


def train_yolo_traffic(
    data_yaml="configs/traffic_dataset.yaml",
    model_name="yolov8n.pt",
    epochs=30,
    imgsz=1280,
    batch=4,
    freeze=10,
    device=None,
    project="runs",
    name="traffic_yolov8_optimized",
    videos=None,
    video_dir="data/raw",
):
    """
    Thực thi quy trình huấn luyện / fine-tuning YOLOv8 tối ưu cho giao thông đô thị.
    """
    from ultralytics import YOLO
    import torch

    print("\n" + "=" * 75)
    print("  QUY TRÌNH HUẤN LUYỆN & FINE-TUNING YOLOV8 CHO NHIỀU VIDEO GIAO THÔNG")
    print("=" * 75)

    # Kiểm tra hoặc tự động tạo dataset nếu chưa có
    train_dir = os.path.join("data", "dataset", "images", "train")
    if not os.path.exists(data_yaml) or not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        print(f"[INFO] Dataset chưa sẵn sàng. Đang tự động quét và trích xuất dữ liệu từ video...")
        v_files = collect_video_files(video_inputs=videos, video_dir=video_dir)
        prepare_dataset_from_videos(video_files=v_files)

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"  Thiết bị huấn luyện: {str(device).upper()}")
    print(f"  Mô hình gốc:         {model_name}")
    print(f"  Độ phân giải:        {imgsz}x{imgsz}")
    print(f"  Số epochs:           {epochs} | Batch size: {batch}")
    print(f"  Đóng băng backbone:  {freeze} layers")
    print("-" * 75)

    model = YOLO(model_name)

    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": project,
        "name": name,
        "exist_ok": True,
        "pretrained": True,
        "freeze": freeze,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 3,
        "weight_decay": 0.0005,
        "mosaic": 1.0,
        "mixup": 0.15,
        "scale": 0.5,
        "degrees": 5.0,
        "fliplr": 0.5,
        "close_mosaic": 5,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "patience": 10,
        "verbose": True,
    }

    start_time = time.time()
    results = model.train(**train_args)
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 75)
    print(f"[HOÀN TẤT HUẤN LUYỆN] Thời gian thực thi: {elapsed_time/60:.2f} phút")
    print("=" * 75)

    save_dir = str(getattr(results, "save_dir", ""))
    best_weight_candidates = [
        os.path.join(save_dir, "weights", "best.pt"),
        os.path.join(project, name, "weights", "best.pt"),
        os.path.join("runs", "detect", name, "weights", "best.pt"),
        os.path.join("runs", "detect", project, name, "weights", "best.pt"),
    ]
    best_weight_path = next((p for p in best_weight_candidates if os.path.exists(p)), None)
    if best_weight_path:
        os.makedirs("weights", exist_ok=True)
        shutil.copy(best_weight_path, "weights/best.pt")
        print(f"[OK] Đã cập nhật trọng số tối ưu nhất vào: weights/best.pt")
        print("     Hệ thống VehicleDetector sẽ tự động nạp trọng số mới này khi khởi động!")

    return results


def validate_model(weights_path="weights/best.pt", data_yaml="configs/traffic_dataset.yaml", imgsz=1280):
    """Đánh giá định lượng mô hình trên tập kiểm chuẩn validation."""
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        if os.path.exists("yolov8n.pt"):
            weights_path = "yolov8n.pt"
        else:
            print(f"[LỖI] Không tìm thấy tệp trọng số: {weights_path}")
            return

    print(f"\n[VALIDATION] Đánh giá mô hình: {weights_path} trên {data_yaml} (imgsz={imgsz})")
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz, verbose=True)

    print("-" * 70)
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("-" * 70)
    return metrics


def export_model(weights_path="weights/best.pt", export_format="onnx", imgsz=1280):
    """Xuất mô hình sang định dạng tối ưu tốc độ ONNX / TorchScript."""
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        weights_path = "yolov8n.pt"

    print(f"\n[EXPORT] Đang xuất mô hình {weights_path} sang định dạng {export_format.upper()}...")
    model = YOLO(weights_path)
    export_path = model.export(format=export_format, imgsz=imgsz, dynamic=True)
    print(f"[OK] Đã xuất mô hình thành công: {export_path}")
    return export_path


def main():
    parser = argparse.ArgumentParser(
        description="Bộ Công Cụ Huấn Luyện & Tối Ưu Hóa Nhận Dạng Phương Tiện Đa Video (UTH)"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Danh sách các tệp video bất kỳ dùng để trích xuất dữ liệu và huấn luyện",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="data/raw",
        help="Thư mục chứa các tệp video (mặc định: data/raw)",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Chỉ thực hiện trích xuất khung hình và gán nhãn giả từ video",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=40,
        help="Số khung hình tối đa trích xuất cho mỗi video (mặc định: 40)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=25,
        help="Bước nhảy khung hình khi trích xuất (mặc định: 25 frames ~ 1s)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/traffic_dataset.yaml",
        help="Đường dẫn tệp cấu hình dataset yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Mô hình khởi tạo (yolov8n.pt, yolov8s.pt hoặc weights/best.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Số lượng epochs huấn luyện (mặc định: 30)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Độ phân giải ảnh huấn luyện (mặc định: 1280)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Kích thước batch size (mặc định: 4)",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Số lớp backbone đóng băng (mặc định: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị ('cpu', '0', hoặc None để tự động)",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Chỉ chạy đánh giá định lượng trên tập validation",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        choices=["onnx", "torchscript"],
        help="Xuất mô hình sang định dạng tối ưu",
    )

    args = parser.parse_args()

    # Thu thập danh sách video
    v_files = collect_video_files(video_inputs=args.videos, video_dir=args.video_dir)

    if args.prepare_data:
        prepare_dataset_from_videos(
            video_files=v_files,
            frame_step=args.frame_step,
            max_frames_per_video=args.max_frames_per_video,
            base_model=args.model,
        )
        return

    if args.val_only:
        validate_model(weights_path=args.model, data_yaml=args.data, imgsz=args.imgsz)
        return

    if args.export:
        export_model(weights_path=args.model, export_format=args.export, imgsz=args.imgsz)
        return

    # Huấn luyện mô hình
    train_yolo_traffic(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        device=args.device,
        videos=args.videos,
        video_dir=args.video_dir,
    )


if __name__ == "__main__":
    main()
