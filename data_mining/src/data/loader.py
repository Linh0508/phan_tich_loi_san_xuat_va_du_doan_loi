# src/data/loader.py
import pandas as pd
import yaml
import os

def load_config(config_path="configs/params.yaml"):
    """Đọc file cấu hình yaml với bảng mã utf-8"""
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_raw_data(config):
    """Đọc dữ liệu thô và tự động xử lý đường dẫn"""
    path = config['data']['raw_path']
    
    # KIỂM TRA: Nếu không thấy đường dẫn, thử lùi lại 1 cấp (dành cho Notebook)
    if not os.path.exists(path):
        alt_path = os.path.join("..", path)
        if os.path.exists(alt_path):
            path = alt_path
    
    if not os.path.exists(path):
        # In ra thư mục hiện tại để bạn dễ kiểm tra lỗi
        current_dir = os.getcwd()
        raise FileNotFoundError(f"Không tìm thấy file tại: {path}. Thư mục hiện tại là: {current_dir}")
    
    df = pd.read_csv(path)
    print(f"✅ Đã tải dữ liệu thành công từ: {path}")
    return df

def validate_data(df, config):
    """Kiểm tra xem các cột quan trọng có tồn tại không"""
    required_columns = config['features']['numerical'] + [config['features']['target']]
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        print(f"❌ Thiếu các cột sau: {missing}")
        return False
    print("✅ Kiểm tra định dạng dữ liệu: Hợp lệ.")
    return True