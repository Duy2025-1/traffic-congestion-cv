"""
SCRIPT HUAN LUYEN & FINE-TUNING MO HINH NHAN DANG PHUONG TIEN (TV5)
Hoc phan: Xu ly anh va Thi giac may tinh (121036) - UTH
Du an: He thong Giam sat va Danh gia Un tac Giao thong Do thi

Cac tinh nang toi uu hoa:
1. Chuan bi tap du lieu tu dong (--prepare-data): Trich xuat frame tu video va gan nhan 4 lop chuan PCU.
2. Sieu tham so toi uu cho mat do giao thong cao (High Density Traffic):
   - imgsz=1280 (giu chi tiet xe nho o xa chan troi).
   - Augmentation: Mosaic (1.0) + Mixup (0.15) chong che khuat (occlusion).
   - Toi uu hoa: AdamW, Warmup 3 epochs, Cosine LR scheduler.
   - Dieu chinh ham mat mat: Tang trong so Box (7.5) va DFL (1.5) de dinh vi khung chinh xac.
   - Dong bang Backbone (--freeze=10) giup hoi tu nhanh va toi uu tai nguyen tinh toan.
3. Tu dong luu trong so tot nhat vao weights/best.pt de VehicleDetector nap tuc thi.
4. Danh gia kiem chuan (--val-only) va Xuat mo hinh (--export onnx).
"""

import os
import sys
import shutil
import argparse
import time
import glob
import cv2
import numpy as np

# Cau hinh encoding UTF-8 tren Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Anh xa tu COCO sang 4 lop phuong tien giao thong chuan hoa
COCO_TO_TRAFFIC = {
    1: 0,  # bicycle -> motorcycle (xe dap/xe may dien)
    3: 0,  # motorcycle -> motorcycle (0)
    2: 1,  # car -> car (1)
    5: 2,  # bus -> bus (2)
    7: 3   # truck -> truck (3)
}

CLASS_NAMES = ["motorcycle", "car", "bus", "truck"]


def prepare_dataset_from_videos(video_dir='data/raw', output_dir='data/dataset', frame_step=25, max_frames_per_video=40):
    """
    Tu dong trich xuat frame tu cac video thuc te va sinh nhan chuan YOLO (pseudo-labeling)
    phuc vu viec fine-tuning ngay lap tuc.
    """
    from ultralytics import YOLO

    print('=' * 75)
    print('  CHUAN BI TAP DU LIEU HUAN LUYEN (DATASET PREPARATION)')
    print('=' * 75)


    os.makedirs(str(output_dir), exist_ok=True)
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not video_files:
        print(f'[LOI] Khong tim thay file video nao trong: {video_dir}')
        return False

    # Khoi tao thu muc chuan cua YOLO
    train_img_dir = os.path.join(output_dir, 'images', 'train')
    val_img_dir = os.path.join(output_dir, 'images', 'val')
    train_lbl_dir = os.path.join(output_dir, 'labels', 'train')
    val_lbl_dir = os.path.join(output_dir, 'labels', 'val')

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    print('[INFO] Dang nap mo hinh baseline de tao nhan chuan...', flush=True)
    baseline_model = YOLO('yolov8n.pt')

    total_extracted = 0
    total_annotations = 0

    for vid_path in video_files:
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        frame_idx = 0
        extracted_for_vid = 0

        while cap.isOpened() and extracted_for_vid < max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_step != 0:
                continue

            h, w = frame.shape[:2]
            
            # Phan chia 80% train, 20% validation
            is_val = (extracted_for_vid % 5 == 0)
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

            file_stem = f'{vid_name}_f{frame_idx:05d}'
            img_out = os.path.join(target_img_dir, f'{file_stem}.jpg')
            lbl_out = os.path.join(target_lbl_dir, f'{file_stem}.txt')

            cv2.imwrite(img_out, frame)

            # Du doan bounding box o do phan giai cao 1280p
            res = baseline_model(
                frame,
                classes=[1, 2, 3, 5, 7],
                conf=0.20,
                iou=0.45,
                imgsz=1280,
                verbose=False
            )[0]

            label_lines = []
            for box in res.boxes:
                coco_cls = int(box.cls[0].item())
                if coco_cls not in COCO_TO_TRAFFIC:
                    continue

                traffic_cls = COCO_TO_TRAFFIC[coco_cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Quy doi sang toa do chuan YOLO [x_center, y_center, width, height] chuan hoa [0, 1]
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                bx = (x1 + x2) / (2.0 * w)
                by = (y1 + y2) / (2.0 * h)

                label_lines.append(f'{traffic_cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}')
                total_annotations += 1

            with open(lbl_out, 'w', encoding='utf-8') as f_lbl:
                f_lbl.write('\n'.join(label_lines) + '\n')

            extracted_for_vid += 1
            total_extracted += 1

        cap.release()
        print(f'  -> {vid_name}: da trich xuat {extracted_for_vid} frames va tao nhan.')

    print('-' * 75)
    print(f'[HOAN TAT] Tong cong da tao {total_extracted} anh va {total_annotations} nhan bounding box.')
    print(f'  - Train: {len(os.listdir(train_img_dir))} anh | Val: {len(os.listdir(val_img_dir))} anh')
    print(f'[OK] Dataset san sang de huan luyen tai: {output_dir}')
    print('=' * 75)
    return True


def train_yolo_traffic(
    data_yaml='configs/traffic_dataset.yaml',
    model_name='yolov8n.pt',
    epochs=30,
    imgsz=1280,
    batch=4,
    freeze=10,
    device=None,
    project='runs',
    name='traffic_yolov8_optimized'
):
    """
    Thuc thi quy trinh huan luyen / fine-tuning YOLOv8 voi bo sieu tham so toi uu hoa.
    """
    from ultralytics import YOLO
    import torch

    print('\n' + '=' * 75)
    print('  QUY TRINH HUAN LUYEN & FINE-TUNING YOLOV8 TOI UU CHO GIAO THONG')
    print('=' * 75)

    if not os.path.exists(data_yaml):
        print(f'[CANH BAO] Khong tim thay file cau hinh dataset: {data_yaml}')
        print('[HUONG DAN] Dang tu dong tao tap du lieu tu cac video trong data/raw/...')
        prepare_dataset_from_videos()

    # Xacdinh thiet bi
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'

    print(f'  Thiet bi huan luyen: {str(device).upper()}')
    print(f'  Mo hinh goc: {model_name}')
    print(f'  Do phan giai (imgsz): {imgsz}')
    print(f'  So epochs: {epochs} | Batch size: {batch}')
    print(f'  Dong bang backbone: {freeze} layers dau')
    print('-' * 75)

    model = YOLO(model_name)

    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': True,
        'pretrained': True,
        'freeze': freeze,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'warmup_epochs': 3,
        'weight_decay': 0.0005,
        'mosaic': 1.0,
        'mixup': 0.15,
        'scale': 0.5,
        'degrees': 5.0,
        'fliplr': 0.5,
        'close_mosaic': 5,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'patience': 10,
        'verbose': True
    }

    start_time = time.time()
    results = model.train(**train_args)
    elapsed_time = time.time() - start_time

    print('\n' + '=' * 75)
    print(f'[HOAN TAT HUAN LUYEN] Thoi gian thuc thi: {elapsed_time/60:.2f} phut')
    print('=' * 75)

    save_dir = str(getattr(results, 'save_dir', ''))
    best_weight_candidates = [
        os.path.join(save_dir, 'weights', 'best.pt'),
        os.path.join(project, name, 'weights', 'best.pt'),
        os.path.join('runs', 'detect', name, 'weights', 'best.pt'),
        os.path.join('runs', 'detect', project, name, 'weights', 'best.pt')
    ]
    best_weight_path = next((p for p in best_weight_candidates if os.path.exists(p)), None)
    if best_weight_path:
        os.makedirs('weights', exist_ok=True)
        shutil.copy(best_weight_path, 'weights/best.pt')
        print(f'[OK] Da sao chep trong so toi uu nhat vao: weights/best.pt')
        print('     He thong VehicleDetector se tu dong nap trong so moi nay khi khoi dong!')
    else:
        print('[THONG TIN] Mo hinh da duoc huan luyen va luu tai:', save_dir)

    return results


def validate_model(weights_path='weights/best.pt', data_yaml='configs/traffic_dataset.yaml', imgsz=1280):
    """Danh gia dinh luong mo hinh tren tap kiem chuan validation."""
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        if os.path.exists('yolov8n.pt'):
            weights_path = 'yolov8n.pt'
        else:
            print(f'[LOI] Khong tim thay trong so: {weights_path}')
            return

    print(f'\n[VALIDATION] Danh gia mo hinh: {weights_path} tren {data_yaml} (imgsz={imgsz})')
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz, verbose=True)

    print('-' * 70)
    print(f'  mAP@50:    {metrics.box.map50:.4f}')
    print(f'  mAP@50-95: {metrics.box.map:.4f}')
    print(f'  Precision: {metrics.box.mp:.4f}')
    print(f'  Recall:    {metrics.box.mr:.4f}')
    print('-' * 70)
    return metrics


def export_model(weights_path='weights/best.pt', export_format='onnx', imgsz=1280):
    """Xuat mo hinh sang dinh dang toi uu toc do ONNX / TorchScript."""
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        weights_path = 'yolov8n.pt'

    print(f'\n[EXPORT] Dang xuat mo hinh {weights_path} sang dinh dang {export_format.upper()}...')
    model = YOLO(weights_path)
    export_path = model.export(format=export_format, imgsz=imgsz, dynamic=True)
    print(f'[OK] Da xuat mo hinh thanh cong: {export_path}')
    return export_path


def main():
    parser = argparse.ArgumentParser(description='He thong Huan luyen & Toi uu hoa Nhan dang Phuong tien (TV5)')
    parser.add_argument('--prepare-data', action='store_true', help='Tu dong tao dataset anh & nhan tu video data/raw/')
    parser.add_argument('--data', type=str, default='configs/traffic_dataset.yaml', help='Duong dan file cau hinh dataset yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Mo hinh khoi tao (yolov8n.pt, yolov8s.pt)')
    parser.add_argument('--epochs', type=int, default=30, help='So luong epochs huan luyen (mac dinh: 30)')
    parser.add_argument('--imgsz', type=int, default=1280, help='Do phan giai anh huan luyen (mac dinh: 1280)')
    parser.add_argument('--batch', type=int, default=4, help='Kich thuoc batch size (mac dinh: 4)')
    parser.add_argument('--freeze', type=int, default=10, help='So lop backbone dong bang (mac dinh: 10)')
    parser.add_argument('--device', type=str, default=None, help='Thiet bi (\'cpu\', \'0\', None de tu dong)')
    parser.add_argument('--val-only', action='store_true', help='Chi chay danh gia tren tap validation')
    parser.add_argument('--export', type=str, default=None, choices=['onnx', 'torchscript'], help='Xuat mo hinh sang dinh dang toi uu')

    args = parser.parse_args()

    if args.prepare_data:
        prepare_dataset_from_videos()
        return

    if args.val_only:
        weights = 'weights/best.pt' if os.path.exists('weights/best.pt') else args.model
        validate_model(weights_path=weights, data_yaml=args.data, imgsz=args.imgsz)
        return

    if args.export:
        weights = 'weights/best.pt' if os.path.exists('weights/best.pt') else args.model
        export_model(weights_path=weights, export_format=args.export, imgsz=args.imgsz)
        return

    # Mac dinh: Chay quy trinh huan luyen
    train_yolo_traffic(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        device=args.device
    )


if __name__ == '__main__':
    main()
