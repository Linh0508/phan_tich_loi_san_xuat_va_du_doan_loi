# src/models/semi_supervised.py
import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier

def prepare_semi_supervised_data(y_train, fraction=0.1, seed=42):
    """Giả lập kịch bản thiếu nhãn (Tiêu chí F)"""
    rng = np.random.RandomState(seed)
    random_unlabeled_points = rng.rand(y_train.shape[0]) > fraction
    
    y_semi = np.copy(y_train)
    y_semi[random_unlabeled_points] = -1
    return y_semi

# Đổi tên hàm này cho khớp với Notebook
def train_self_training(X_train, y_semi, config):
    """Huấn luyện Self-training để tận dụng dữ liệu chưa có nhãn"""
    base_model = RandomForestClassifier(
        n_estimators=50, 
        class_weight='balanced',
        random_state=config['split']['random_state']
    )
    # Threshold 0.75: Chỉ tin tưởng các pseudo-label có xác suất > 75%
    self_training_model = SelfTrainingClassifier(base_model, threshold=0.75)
    self_training_model.fit(X_train, y_semi)
    return self_training_model