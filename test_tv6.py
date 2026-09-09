"""
File kiểm thử TV6 - Đánh giá định lượng.

Các nội dung được kiểm thử:
1. Tính các chỉ số Accuracy, Precision, Recall, F1-score.
2. Kiểm tra dữ liệu không hợp lệ.
3. Tạo và lưu ma trận nhầm lẫn.
4. Tính sai số vận tốc MAE.
5. Kiểm tra các trường hợp lỗi của dữ liệu vận tốc.

Lưu ý:
Dữ liệu trong file này chỉ dùng để kiểm thử chức năng.
Không phải dữ liệu thực nghiệm chính thức của nhóm.
"""

from pathlib import Path

import numpy as np
import pytest

from evaluation.metrics import (
    compute_classification_metrics,
    compute_speed_mae,
    plot_confusion_matrix,
)


# ============================================================
# 1. KIỂM TRA TÍNH CÁC CHỈ SỐ PHÂN LOẠI
# ============================================================

def test_tinh_chi_so_phan_loai():
    """
    Kiểm tra hàm tính Accuracy, Precision, Recall và F1-score.
    """

    # Bốn mức độ ùn tắc
    cac_muc_do = [
        "1 - Thong thoang",
        "2 - Binh thuong",
        "3 - Un u",
        "4 - Tac nghen",
    ]

    # Nhãn thực tế
    nhan_thuc_te = [
        cac_muc_do[0],
        cac_muc_do[1],
        cac_muc_do[2],
        cac_muc_do[3],
    ]

    # Nhãn dự đoán
    nhan_du_doan = [
        cac_muc_do[0],
        cac_muc_do[1],
        cac_muc_do[2],
        cac_muc_do[3],
    ]

    # Gọi hàm cần kiểm thử
    ket_qua = compute_classification_metrics(
        nhan_thuc_te,
        nhan_du_doan,
    )

    # Khi dự đoán hoàn toàn chính xác,
    # tất cả các chỉ số phải bằng 1.
    assert ket_qua["accuracy"] == 1.0
    assert ket_qua["precision_macro"] == 1.0
    assert ket_qua["recall_macro"] == 1.0
    assert ket_qua["f1_macro"] == 1.0
    assert ket_qua["precision_weighted"] == 1.0
    assert ket_qua["recall_weighted"] == 1.0
    assert ket_qua["f1_weighted"] == 1.0


# ============================================================
# 2. KIỂM TRA KHI SỐ LƯỢNG NHÃN KHÁC NHAU
# ============================================================

def test_nhan_thuc_te_va_du_doan_khac_so_luong():
    """
    Kiểm tra trường hợp Ground Truth và Prediction
    có số lượng phần tử khác nhau.
    """

    nhan_thuc_te = [
        "1 - Thong thoang",
        "2 - Binh thuong",
        "3 - Un u",
    ]

    nhan_du_doan = [
        "1 - Thong thoang",
        "2 - Binh thuong",
    ]

    # Phải phát sinh lỗi ValueError
    with pytest.raises(ValueError):
        compute_classification_metrics(
            nhan_thuc_te,
            nhan_du_doan,
        )


# ============================================================
# 3. KIỂM TRA KHI DỮ LIỆU RỖNG
# ============================================================

def test_du_lieu_phan_loai_rong():
    """
    Kiểm tra trường hợp dữ liệu đánh giá bị rỗng.
    """

    with pytest.raises(ValueError):
        compute_classification_metrics(
            [],
            [],
        )


# ============================================================
# 4. KIỂM TRA MA TRẬN NHẦM LẪN
# ============================================================

def test_tao_ma_tran_nham_lan(tmp_path):
    """
    Kiểm tra hàm tạo và lưu ma trận nhầm lẫn.
    """

    cac_muc_do = [
        "1 - Thong thoang",
        "2 - Binh thuong",
        "3 - Un u",
        "4 - Tac nghen",
    ]

    nhan_thuc_te = [
        cac_muc_do[0],
        cac_muc_do[1],
        cac_muc_do[2],
        cac_muc_do[3],
    ]

    nhan_du_doan = [
        cac_muc_do[0],
        cac_muc_do[1],
        cac_muc_do[1],
        cac_muc_do[3],
    ]

    # Lưu file tạm để không làm thay đổi
    # file kết quả chính của project.
    duong_dan = tmp_path / "ma_tran_nham_lan.png"

    ket_qua = plot_confusion_matrix(
        nhan_thuc_te,
        nhan_du_doan,
        cac_muc_do,
        duong_dan,
    )

    # Kiểm tra đường dẫn trả về
    assert ket_qua == duong_dan

    # Kiểm tra hình ảnh đã được tạo
    assert duong_dan.exists()

    # Kiểm tra file không rỗng
    assert duong_dan.stat().st_size > 0


# ============================================================
# 5. KIỂM TRA KHI SỐ LƯỢNG NHÃN KHÁC NHAU
# ============================================================

def test_ma_tran_nham_lan_khac_so_luong():
    """
    Kiểm tra hàm ma trận nhầm lẫn khi Ground Truth
    và Prediction có số lượng khác nhau.
    """

    nhan_thuc_te = [
        "1 - Thong thoang",
        "2 - Binh thuong",
    ]

    nhan_du_doan = [
        "1 - Thong thoang",
    ]

    cac_muc_do = [
        "1 - Thong thoang",
        "2 - Binh thuong",
        "3 - Un u",
        "4 - Tac nghen",
    ]

    with pytest.raises(ValueError):
        plot_confusion_matrix(
            nhan_thuc_te,
            nhan_du_doan,
            cac_muc_do,
        )


# ============================================================
# 6. KIỂM TRA DANH SÁCH MỨC ĐỘ ÙN TẮC RỖNG
# ============================================================

def test_ma_tran_nham_lan_khong_co_muc_do():
    """
    Kiểm tra trường hợp danh sách các mức độ ùn tắc rỗng.
    """

    nhan_thuc_te = [
        "1 - Thong thoang",
    ]

    nhan_du_doan = [
        "1 - Thong thoang",
    ]

    with pytest.raises(ValueError):
        plot_confusion_matrix(
            nhan_thuc_te,
            nhan_du_doan,
            [],
        )


# ============================================================
# 7. KIỂM TRA TÍNH MAE CỦA VẬN TỐC
# ============================================================

def test_tinh_sai_so_van_toc_mae():
    """
    Kiểm tra công thức MAE.

    Ví dụ:
        Vận tốc thực tế  : 20, 30, 40
        Vận tốc dự đoán  : 22, 27, 43

    Sai số:
        |20 - 22| = 2
        |30 - 27| = 3
        |40 - 43| = 3

    MAE = (2 + 3 + 3) / 3 = 2.6667
    """

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

    mae = compute_speed_mae(
        van_toc_thuc_te,
        van_toc_du_doan,
    )

    # So sánh với giá trị mong đợi
    assert np.isclose(
        mae,
        2.6666666667,
        atol=1e-6,
    )


# ============================================================
# 8. KIỂM TRA MAE KHI DỰ ĐOÁN HOÀN TOÀN ĐÚNG
# ============================================================

def test_mae_bang_khong_khi_du_doan_dung():
    """
    Nếu vận tốc thực tế và vận tốc dự đoán giống nhau
    thì MAE phải bằng 0.
    """

    van_toc_thuc_te = [
        20,
        30,
        40,
        50,
    ]

    van_toc_du_doan = [
        20,
        30,
        40,
        50,
    ]

    mae = compute_speed_mae(
        van_toc_thuc_te,
        van_toc_du_doan,
    )

    assert mae == 0.0


# ============================================================
# 9. KIỂM TRA MAE KHI DỮ LIỆU RỖNG
# ============================================================

def test_mae_du_lieu_rong():
    """
    Kiểm tra trường hợp dữ liệu vận tốc bị rỗng.
    """

    with pytest.raises(ValueError):
        compute_speed_mae(
            [],
            [],
        )


# ============================================================
# 10. KIỂM TRA MAE KHI KÍCH THƯỚC KHÁC NHAU
# ============================================================

def test_mae_khac_kich_thuoc():
    """
    Kiểm tra trường hợp số lượng vận tốc thực tế
    và vận tốc dự đoán không bằng nhau.
    """

    van_toc_thuc_te = [
        20,
        30,
        40,
    ]

    van_toc_du_doan = [
        20,
        30,
    ]

    with pytest.raises(ValueError):
        compute_speed_mae(
            van_toc_thuc_te,
            van_toc_du_doan,
        )


# ============================================================
# 11. KIỂM TRA MAE KHI CÓ DỮ LIỆU KHÔNG HỢP LỆ
# ============================================================

def test_mae_co_du_lieu_khong_hop_le():
    """
    Kiểm tra trường hợp vận tốc chứa NaN hoặc vô cực.
    """

    van_toc_thuc_te = [
        20,
        np.nan,
        40,
    ]

    van_toc_du_doan = [
        21,
        30,
        39,
    ]

    with pytest.raises(ValueError):
        compute_speed_mae(
            van_toc_thuc_te,
            van_toc_du_doan,
        )


# ============================================================
# 12. KIỂM TRA KHI SAI SỐ VẬN TỐC ĐỀU NHAU
# ============================================================

def test_mae_sai_so_deu_nhau():
    """
    Kiểm tra một trường hợp MAE đơn giản:

        Thực tế : 10, 20, 30
        Dự đoán : 12, 22, 32

    Mỗi mẫu sai 2 đơn vị nên MAE = 2.
    """

    van_toc_thuc_te = [
        10,
        20,
        30,
    ]

    van_toc_du_doan = [
        12,
        22,
        32,
    ]

    mae = compute_speed_mae(
        van_toc_thuc_te,
        van_toc_du_doan,
    )

    assert mae == 2.0