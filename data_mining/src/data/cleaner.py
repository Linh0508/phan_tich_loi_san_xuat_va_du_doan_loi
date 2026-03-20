import pandas as pd

def clean_data(df):
    """Làm sạch dữ liệu thô"""
    # 1. Loại bỏ các cột không cần thiết cho huấn luyện
    # UDI: ID dòng, Product ID: Mã sản phẩm (đã có cột Type đại diện cho chất lượng)
    cols_to_drop = ['UDI', 'Product ID']
    df_cleaned = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 2. Kiểm tra và loại bỏ dòng trùng lặp
    df_cleaned = df_cleaned.drop_duplicates()
    
    # 3. Xử lý giá trị thiếu (nếu có)
    df_cleaned = df_cleaned.dropna()
    
    print(f"✅ Đã làm sạch dữ liệu. Kích thước mới: {df_cleaned.shape}")
    return df_cleaned