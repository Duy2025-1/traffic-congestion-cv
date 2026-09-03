from modules.congestion_evaluator import TrafficCongestionEvaluator

def main():
    print("=== Traffic Congestion Estimation System - UTH ===")
    evaluator = TrafficCongestionEvaluator()
    sample_result = evaluator.compute_tci(occupancy_ratio=0.6, avg_speed=12.0, pcu_count=30.0)
    print("Pipeline khởi động thành công. Kết quả test mẫu:", sample_result)

if __name__ == "__main__":
    main()
