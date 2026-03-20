from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def train_classification(X_train, y_train, config):
    """
    Huấn luyện các mô hình phân lớp lỗi máy.
    Đáp ứng tiêu chí D: >= 2 baseline + 1 cải tiến.
    """
    # Baseline 1: Logistic Regression (Mô hình đơn giản)
    base_model_1 = LogisticRegression(random_state=config['split']['random_state'])
    base_model_1.fit(X_train, y_train)
    
    # Baseline 2: RandomForest
    base_model_2 = RandomForestClassifier(
        n_estimators=100, 
        random_state=config['split']['random_state'],
        class_weight='balanced' # Xử lý imbalance (Tiêu chí E)
    )
    base_model_2.fit(X_train, y_train)
    
    # Mô hình cải tiến: XGBoost
    improved_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=config['split']['random_state']
    )
    improved_model.fit(X_train.values, y_train.values)
    
    return base_model_1, base_model_2, improved_model

def train_regression(X_train, y_train, config):
    """Dự đoán Tool wear [min] (Yêu cầu đề tài 16)"""
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train.values, y_train.values)
    return model