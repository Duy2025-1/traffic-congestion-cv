"""
FILE CẤU HÌNH HỆ THỐNG - UTH TRAFFIC MONITORING
Quản lý bởi: TV1 (Leader)
"""

WEIGHT_OCCUPANCY = 0.4    # w1: Tỷ lệ chiếm dụng (TV3)
WEIGHT_SPEED = 0.4        # w2: Suy giảm vận tốc (TV4)
WEIGHT_PCU = 0.2          # w3: Mật độ tải trọng PCU (TV5)

LEVEL_THRESHOLDS = {
    "FREE_FLOW": 0.30,
    "MODERATE": 0.55,
    "HEAVY": 0.75
}

FREE_FLOW_SPEED_KMH = 40.0
MAX_PCU_CAPACITY = 50.0

PCU_WEIGHTS = {
    "motorcycle": 0.33,
    "car": 1.0,
    "bus": 2.5,
    "truck": 2.5
}
