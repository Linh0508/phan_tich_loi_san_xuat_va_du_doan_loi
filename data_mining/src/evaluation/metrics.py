# src/evaluation/metrics.py
from sklearn.metrics import f1_score, precision_recall_curve, auc

def calculate_pr_auc(y_true, y_probs):
    """Tính PR-AUC - Metric quan trọng nhất cho dữ liệu mất cân bằng (Imbalance)"""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)

def evaluate_model(model, X_test, y_test):
    """Đánh giá mô hình và trả về F1 và PR-AUC"""
    y_pred = model.predict(X_test)
    # Loại bỏ nhãn -1 nếu có trong y_test (thường y_test luôn có đủ nhãn)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred, average='binary')
    pr_auc = calculate_pr_auc(y_test, y_probs)
    
    return f1, pr_auc