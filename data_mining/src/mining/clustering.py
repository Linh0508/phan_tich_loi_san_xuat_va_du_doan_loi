# src/mining/clustering.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_clustering(df, config):
    """Phân cụm các trạng thái vận hành của máy"""
    # Lấy các cột cảm biến để phân cụm
    features = config['features']['numerical']
    X = df[features]
    
    n_clusters = config.get('mining', {}).get('n_clusters', 3)
    
    # Huấn luyện KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=config['split']['random_state'], n_init=10)
    clusters = kmeans.fit_predict(X)
    
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    # Tính điểm Silhouette để đánh giá chất lượng cụm
    score = silhouette_score(X, clusters)
    print(f"✅ Phân cụm hoàn tất. Silhouette Score: {score:.4f}")
    
    return df_clustered, kmeans