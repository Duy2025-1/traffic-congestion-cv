"""
Các hàm đánh giá định lượng cho TV6 - Đánh giá mức độ ùn tắc giao thông.

Nhiệm vụ chính:
- Tính Accuracy, Precision, Recall và F1-score.
- Vẽ và lưu ma trận nhầm lẫn.
- Tính sai số tuyệt đối trung bình (MAE) của vận tốc.
"""

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# Đường dẫn mặc định để lưu ma trận nhầm lẫn
DUONG_DAN_MA_TRAN_NHAM_LAN = Path(
    "docs/figures/confusion_matrix.png"
)


def compute_classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
) -> Mapping[str, float]:
    """
    Tính các chỉ số đánh giá cho bài toán phân loại mức độ ùn tắc.

    Tham số:
        y_true:
            Nhãn thực tế (Ground Truth).

        y_pred:
            Nhãn do hệ thống dự đoán (Prediction).

    Kết quả trả về:
        Từ điển gồm:
        - Accuracy
        - Precision macro
        - Recall macro
        - F1 macro
        - Precision weighted
        - Recall weighted
        - F1 weighted
    """

    # Kiểm tra số lượng nhãn thực tế và nhãn dự đoán
    if len(y_true) != len(y_pred):
        raise ValueError(
            "Nhãn thực tế và nhãn dự đoán phải có cùng số lượng phần tử."
        )

    # Kiểm tra dữ liệu có rỗng hay không
    if len(y_true) == 0:
        raise ValueError(
            "Dữ liệu nhãn thực tế và nhãn dự đoán không được để trống."
        )

    # Tính các chỉ số đánh giá
    do_chinh_xac = accuracy_score(y_true, y_pred)

    do_chinh_xac_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    do_bao_phu_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    do_chinh_xac_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    do_bao_phu_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(do_chinh_xac),
        "precision_macro": float(do_chinh_xac_macro),
        "recall_macro": float(do_bao_phu_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(do_chinh_xac_weighted),
        "recall_weighted": float(do_bao_phu_weighted),
        "f1_weighted": float(f1_weighted),
    }


def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    class_names: Sequence,
    output_path: str | Path = DUONG_DAN_MA_TRAN_NHAM_LAN,
) -> Path:
    """
    Vẽ và lưu ma trận nhầm lẫn.

    Hàng:
        Nhãn thực tế.

    Cột:
        Nhãn dự đoán.

    Tham số:
        y_true:
            Nhãn thực tế.

        y_pred:
            Nhãn dự đoán.

        class_names:
            Danh sách 4 mức độ ùn tắc.

        output_path:
            Đường dẫn lưu hình ảnh ma trận nhầm lẫn.
    """

    # Kiểm tra số lượng dữ liệu
    if len(y_true) != len(y_pred):
        raise ValueError(
            "Nhãn thực tế và nhãn dự đoán phải có cùng số lượng phần tử."
        )

    # Kiểm tra danh sách lớp
    if not class_names:
        raise ValueError(
            "Danh sách mức độ ùn tắc không được để trống."
        )

    # Tính ma trận nhầm lẫn
    ma_tran = confusion_matrix(
        y_true,
        y_pred,
        labels=list(class_names),
    )

    # Chuyển đường dẫn về dạng Path
    output_path = Path(output_path)

    # Tạo thư mục nếu chưa tồn tại
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Tạo biểu đồ
    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    hinh_anh = ax.imshow(
        ma_tran,
        interpolation="nearest",
    )

    # Thanh chú thích giá trị
    fig.colorbar(
        hinh_anh,
        ax=ax,
    )

    # Thiết lập tên trục
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Nhãn thực tế",
        xlabel="Nhãn dự đoán",
        title="Ma trận nhầm lẫn mức độ ùn tắc giao thông",
    )

    # Xác định ngưỡng để hiển thị số màu trắng hoặc đen
    nguong = (
        ma_tran.max() / 2.0
        if ma_tran.size
        else 0.0
    )

    # Hiển thị giá trị tại từng ô
    for hang in range(ma_tran.shape[0]):
        for cot in range(ma_tran.shape[1]):
            ax.text(
                cot,
                hang,
                str(ma_tran[hang, cot]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if ma_tran[hang, cot] > nguong
                    else "black"
                ),
            )

    # Căn chỉnh biểu đồ
    fig.tight_layout()

    # Lưu hình ảnh
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    # Đóng biểu đồ sau khi lưu
    plt.close(fig)

    return output_path


def compute_speed_mae(
    y_true_speed: Iterable[float],
    y_pred_speed: Iterable[float],
) -> float:
    """
    Tính sai số tuyệt đối trung bình (MAE) của vận tốc.

    MAE = Trung bình |Vận tốc thực tế - Vận tốc dự đoán|

    Tham số:
        y_true_speed:
            Vận tốc thực tế.

        y_pred_speed:
            Vận tốc do hệ thống dự đoán.

    Kết quả:
        Giá trị MAE.
    """

    # Chuyển dữ liệu về mảng số thực
    van_toc_thuc_te = np.asarray(
        list(y_true_speed),
        dtype=float,
    )

    van_toc_du_doan = np.asarray(
        list(y_pred_speed),
        dtype=float,
    )

    # Kiểm tra dữ liệu rỗng
    if (
        van_toc_thuc_te.size == 0
        or van_toc_du_doan.size == 0
    ):
        raise ValueError(
            "Dữ liệu vận tốc không được để trống."
        )

    # Kiểm tra kích thước
    if van_toc_thuc_te.shape != van_toc_du_doan.shape:
        raise ValueError(
            "Vận tốc thực tế và vận tốc dự đoán "
            "phải có cùng kích thước."
        )

    # Kiểm tra dữ liệu có phải số hữu hạn hay không
    if (
        not np.all(np.isfinite(van_toc_thuc_te))
        or not np.all(np.isfinite(van_toc_du_doan))
    ):
        raise ValueError(
            "Dữ liệu vận tốc phải là các số hữu hạn."
        )

    # Tính sai số tuyệt đối trung bình
    sai_so_tuyet_doi = np.abs(
        van_toc_thuc_te - van_toc_du_doan
    )

    sai_so_trung_binh = np.mean(
        sai_so_tuyet_doi
    )

    return float(sai_so_trung_binh)


if __name__ == "__main__":
    """
    Kiểm tra nhanh các chức năng của TV6.

    Lưu ý:
    Dữ liệu bên dưới chỉ là dữ liệu mẫu để kiểm tra code.
    Đây KHÔNG phải kết quả thực nghiệm chính thức của nhóm.
    """

    # Bốn mức độ ùn tắc giao thông
    cac_muc_do = [
        "1 - Thong thoang",
        "2 - Binh thuong",
        "3 - Un u",
        "4 - Tac nghen",
    ]

    # Nhãn thực tế mẫu
    nhan_thuc_te = [
        cac_muc_do[0],
        cac_muc_do[1],
        cac_muc_do[2],
        cac_muc_do[3],
        cac_muc_do[3],
        cac_muc_do[2],
    ]

    # Nhãn dự đoán mẫu
    nhan_du_doan = [
        cac_muc_do[0],
        cac_muc_do[1],
        cac_muc_do[1],
        cac_muc_do[3],
        cac_muc_do[2],
        cac_muc_do[2],
    ]

    # Tính các chỉ số phân loại
    ket_qua = compute_classification_metrics(
        nhan_thuc_te,
        nhan_du_doan,
    )

    # Vẽ và lưu ma trận nhầm lẫn
    duong_dan_hinh = plot_confusion_matrix(
        nhan_thuc_te,
        nhan_du_doan,
        cac_muc_do,
    )

    # Dữ liệu vận tốc mẫu
    van_toc_thuc_te = [
        20,
        30,
        40,
    ]

    van_toc_du_doan = [
        22,
        27,
        43,
    ]

    # Tính MAE
    mae = compute_speed_mae(
        van_toc_thuc_te,
        van_toc_du_doan,
    )

    # In kết quả
    print("======================================")
    print("      KIỂM TRA TV6 - ĐÁNH GIÁ")
    print("======================================")

    print("Kiểm tra code TV6: THÀNH CÔNG")

    print(
        f"Độ chính xác (Accuracy): "
        f"{ket_qua['accuracy']:.4f}"
    )

    print(
        f"Precision (Macro): "
        f"{ket_qua['precision_macro']:.4f}"
    )

    print(
        f"Recall (Macro): "
        f"{ket_qua['recall_macro']:.4f}"
    )

    print(
        f"F1-score (Macro): "
        f"{ket_qua['f1_macro']:.4f}"
    )

    print(
        f"F1-score (Weighted): "
        f"{ket_qua['f1_weighted']:.4f}"
    )

    print(
        f"Sai số vận tốc MAE: "
        f"{mae:.4f}"
    )

    print(
        f"Ma trận nhầm lẫn được lưu tại: "
        f"{duong_dan_hinh}"
    )

    print("======================================")
    print("Lưu ý: Đây chỉ là dữ liệu kiểm tra mẫu.")
    print("Không phải kết quả thực nghiệm chính thức.")
    print("======================================")