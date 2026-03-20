# src/visualization/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_cluster_comparison(df, config):
    """Vẽ biểu đồ so sánh các cảm biến giữa các cụm"""
    features = config['features']['numerical']
    
    # Chuẩn bị dữ liệu dạng long-format để vẽ
    df_melted = df.melt(id_vars=['Cluster'], value_vars=features, 
                        var_name='Sensor', value_name='Value')
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Sensor', y='Value', hue='Cluster', data=df_melted)
    plt.title("So sánh đặc trưng cảm biến giữa các cụm (Scaled)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_failure_by_cluster(df):
    """Vẽ tỷ lệ lỗi máy thực tế trong từng cụm"""
    failure_rate = df.groupby('Cluster')['Machine failure'].mean() * 100
    
    plt.figure(figsize=(8, 5))
    ax = failure_rate.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
    plt.title("Tỷ lệ lỗi máy thực tế theo từng cụm (%)")
    plt.ylabel("Tỷ lệ lỗi (%)")
    plt.xlabel("Cụm (Cluster)")
    
    # Ghi số liệu lên đầu cột
    for i, v in enumerate(failure_rate):
        ax.text(i - 0.1, v + 0.1, f"{v:.2f}%", fontweight='bold')
    
    plt.show()

def plot_feature_importance(model, feature_names):
    """Vẽ biểu đồ tầm quan trọng đặc trưng"""
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=feature_names)
    feat_importances = feat_importances.sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    feat_importances.plot(kind='barh', color='teal')
    plt.title('Feature Importance Analysis')
    plt.xlabel('Score')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Vẽ ma trận nhầm lẫn"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Failure', 'Failure'],
                yticklabels=['No Failure', 'Failure'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()