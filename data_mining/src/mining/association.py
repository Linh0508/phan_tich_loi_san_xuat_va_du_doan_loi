# src/mining/association.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def get_association_rules(df, config):
    """Tìm luật kết hợp giữa các trạng thái máy và lỗi"""
    df_mining = df.copy()
    
    # 1. Rời rạc hóa: Chuyển các chỉ số cảm biến thành dạng 'Cao/Thấp'
    # Ví dụ: Nếu nhiệt độ > trung bình thì coi là 'High_Temp'
    for col in config['features']['numerical']:
        mean_val = df_mining[col].mean()
        df_mining[col] = df_mining[col].apply(lambda x: f"{col}_High" if x > mean_val else f"{col}_Low")
    
    # 2. One-hot encoding để đưa về dạng 0/1 cho thuật toán Apriori
    # Tập trung vào các cột cảm biến và cột lỗi (Machine failure)
    cols_for_rules = config['features']['numerical'] + ['Machine failure']
    df_encoded = pd.get_dummies(df_mining[cols_for_rules])
    
    # 3. Chạy thuật toán Apriori tìm các tập phổ biến
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
    
    # 4. Tạo luật kết hợp
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    
    # Lọc các luật mà kết quả (consequents) dẫn đến 'Machine failure_1' (có lỗi)
    error_rules = rules[rules['consequents'].apply(lambda x: 'Machine failure_1' in str(x))]
    
    print(f"✅ Tìm thấy {len(error_rules)} luật dẫn đến lỗi máy.")
    return error_rules.sort_values('lift', ascending=False)

def get_specific_failure_rules(df, config, failure_type="HDF"):
    """
    Tìm luật cho một loại lỗi cụ thể (HDF, PWF, OSF, TWF, RNF)
    Đã sửa lỗi kiểu dữ liệu cho mlxtend.
    """
    df_mining = df.copy()
    
    # 1. Rời rạc hóa (Binning)
    numerical_cols = config['features']['numerical']
    for col in numerical_cols:
        mean_val = df_mining[col].mean()
        df_mining[col] = df_mining[col].apply(lambda x: f"{col}_High" if x > mean_val else f"{col}_Low")
    
    # 2. Chọn cột cảm biến và 1 loại lỗi cụ thể
    # Quan trọng: Cột failure_type phải là 0/1 hoặc True/False
    cols_for_rules = numerical_cols + [failure_type]
    df_encoded = pd.get_dummies(df_mining[cols_for_rules])
    
    # 3. ÉP KIỂU SANG BOOLEAN (Để tránh lỗi mlxtend version mới)
    df_encoded = df_encoded.astype(bool)
    
    # 4. Chạy Apriori
    # Nếu vẫn 0 luật, hãy giảm min_support xuống thêm nữa
    frequent_itemsets = apriori(df_encoded, min_support=0.0001, use_colnames=True)
    
    if frequent_itemsets.empty:
        return pd.DataFrame() # Trả về df rỗng nếu không tìm thấy tập phổ biến

    # 5. Tạo luật
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # 6. Lọc luật dẫn đến lỗi (Target là cột failure_type có giá trị True)
    # Sau khi get_dummies và ép bool, cột mục tiêu thường là failure_type_True hoặc failure_type_1
    target_col = f"{failure_type}_True" if f"{failure_type}_True" in df_encoded.columns else failure_type
    
    # Tìm trong consequents những luật dẫn đến lỗi
    specific_rules = rules[rules['consequents'].apply(lambda x: any(target_col in str(item) for item in x))]
    
    return specific_rules.sort_values('lift', ascending=False)

def compare_failure_patterns(df, config):
    """
    Hàm bổ sung: Tự động quét qua tất cả loại lỗi và tổng hợp số lượng luật
    Giúp bạn có cái nhìn tổng quan để so sánh dễ hơn.
    """
    failure_types = ['HDF', 'PWF', 'OSF', 'TWF', 'RNF']
    summary = {}
    
    print("--- ĐANG SO SÁNH PATTERN CỦA CÁC LOẠI LỖI ---")
    for f_type in failure_types:
        try:
            rules = get_specific_failure_rules(df, config, failure_type=f_type)
            summary[f_type] = len(rules)
            print(f"Lỗi {f_type}: Tìm thấy {len(rules)} luật.")
        except Exception as e:
            summary[f_type] = 0
            
    return summary