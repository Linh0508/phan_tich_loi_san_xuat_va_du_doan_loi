# scripts/run_pipeline.py
import sys
import os
import pandas as pd

# Thêm thư mục gốc vào path để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_config, load_raw_data
from src.data.cleaner import clean_data
from src.features.builder import build_features
from src.models.supervised import train_classification
from src.models.semi_supervised import prepare_semi_supervised_data, train_self_training
from src.evaluation.metrics import evaluate_model
from src.evaluation.report import generate_comparison_report

def main():
    print("🚀 Bắt đầu chạy Pipeline Dự đoán lỗi sản xuất (Đề tài 16)...")
    
    # 1. Load Cấu hình & Dữ liệu
    config = load_config("configs/params.yaml")
    df_raw = load_raw_data(config)
    
    # 2. Tiền xử lý & Đặc trưng
    df_cleaned = clean_data(df_raw)
    df_final = build_features(df_cleaned, config)
    
    # Chia dữ liệu Train/Test (Giả định đơn giản cho script)
    train_df = df_final.sample(frac=0.8, random_state=42)
    test_df = df_final.drop(train_df.index)
    
    X_train = train_df.drop(columns=[config['features']['target']])
    y_train = train_df[config['features']['target']]
    X_test = test_df.drop(columns=[config['features']['target']])
    y_test = test_df[config['features']['target']]
    
    # 3. Huấn luyện Giám sát (Supervised) - Toàn bộ nhãn
    print("--- Huấn luyện mô hình Giám sát ---")
    _, rf_full, _ = train_classification(X_train, y_train, config)
    f1_full, auc_full = evaluate_model(rf_full, X_test, y_test)
    
    # 4. Huấn luyện Bán giám sát (Semi-supervised) - Chỉ 10% nhãn
    print("--- Huấn luyện mô hình Bán giám sát (10% nhãn) ---")
    y_semi = prepare_semi_supervised_data(y_train, fraction=0.1)
    
    # Nhánh so sánh: Chỉ học trên 10% nhãn (Supervised-only)
    X_small = X_train[y_semi != -1]
    y_small = y_semi[y_semi != -1]
    from sklearn.ensemble import RandomForestClassifier
    rf_small = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
    rf_small.fit(X_small, y_small)
    f1_small, auc_small = evaluate_model(rf_small, X_test, y_test)
    
    # Nhánh Self-training: Học trên 10% nhãn + 90% ẩn
    rf_semi = train_self_training(X_train, y_semi, config)
    f1_semi, auc_semi = evaluate_model(rf_semi, X_test, y_test)
    
    # 5. Tổng hợp báo cáo
    metrics = {
        "Supervised (100% labels)": {"F1": f1_full, "PR-AUC": auc_full},
        "Supervised (10% labels)": {"F1": f1_small, "PR-AUC": auc_small},
        "Semi-Supervised (10% labels)": {"F1": f1_semi, "PR-AUC": auc_semi}
    }
    
    report = generate_comparison_report(metrics)
    print("\n📊 KẾT QUẢ SO SÁNH MÔ HÌNH:")
    print(report)
    print("\n✅ Pipeline hoàn tất! Kết quả đã được lưu vào outputs/tables/model_comparison.csv")

if __name__ == "__main__":
    main()