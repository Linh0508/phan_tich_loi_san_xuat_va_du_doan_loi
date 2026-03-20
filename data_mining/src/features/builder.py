from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def build_features(df, config):
    """Tạo và biến đổi đặc trưng"""
    df_feat = df.copy()
    
    # 1. Encode biến phân loại 'Type' (L=0, M=1, H=2)
    # Trong bài toán bảo trì, Type (chất lượng máy) rất quan trọng
    le = LabelEncoder()
    if 'Type' in df_feat.columns:
        df_feat['Type'] = le.fit_transform(df_feat['Type'])
    
    # 2. Tạo đặc trưng mới: Hiệu số nhiệt độ (Temperature Difference)
    # Đây là insight kỹ thuật: Sự chênh lệch giữa nhiệt độ máy và môi trường thường gây lỗi
    df_feat['Temp_Diff'] = df_feat['Process temperature [K]'] - df_feat['Air temperature [K]']
    
    # 3. Chuẩn hóa dữ liệu (Scaling) cho các cột numerical
    scaler = StandardScaler()
    num_cols = config['features']['numerical'] + ['Temp_Diff']
    df_feat[num_cols] = scaler.fit_transform(df_feat[num_cols])
    
    return df_feat