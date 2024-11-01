import pandas as pd
import numpy as np
from scipy.stats import mode

# 1. Chuẩn bị dữ liệu từ file `iris.data`
data = pd.read_csv('./iris/iris.data', header=None)
X = data.iloc[:, :-1].values  # Các cột đặc trưng
y_true = data.iloc[:, -1].values  # Cột nhãn

# Chuyển đổi nhãn từ tên loại hoa thành số để tiện xử lý
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(y_true)

# 2. Hàm tính khoảng cách Euclidean
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# 3. Khởi tạo ngẫu nhiên các centroid
def initialize_centroids(X, k):
    np.random.seed(0)
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

# 4. Phân cụm cho các điểm dữ liệu
def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)

# 5. Cập nhật vị trí các centroid
def update_centroids(X, clusters, k):
    centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        centroid = cluster_points.mean(axis=0) if len(cluster_points) > 0 else X[np.random.choice(X.shape[0])]
        centroids.append(centroid)
    return np.array(centroids)

# 6. Hàm thuật toán K-means
def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# 7. Phân cụm với K-means (K=3 vì bộ dữ liệu Iris có 3 loại hoa)
k = 3
y_pred, centroids = kmeans(X, k)

# 8. Ánh xạ các cụm vào nhãn thật
def map_clusters_to_labels(y_true, y_pred):
    labels = np.zeros_like(y_pred)
    for i in range(k):
        mask = (y_pred == i)
        labels[mask] = mode(y_true[mask])[0]
    return labels

y_pred_mapped = map_clusters_to_labels(y_true, y_pred)

# 9. Tính các chỉ số đánh giá

# F1-score tính thủ công
def f1_score_manual(y_true, y_pred):
    TP = sum((y_true == y_pred) & (y_pred == 1))
    FP = sum((y_true != y_pred) & (y_pred == 1))
    FN = sum((y_true != y_pred) & (y_pred == 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# RAND index tính thủ công
def rand_index_manual(y_true, y_pred):
    tp_fp = np.sum([np.sum(y_pred == i) ** 2 for i in range(k)])
    tp_fn = np.sum([np.sum(y_true == i) ** 2 for i in range(k)])
    tp = np.sum((y_true == y_pred).astype(int)) ** 2
    rand_index = tp / (tp_fp + tp_fn - tp) if (tp_fp + tp_fn - tp) > 0 else 0
    return rand_index

# NMI tính thủ công
def nmi_manual(y_true, y_pred):
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    total = len(y_true)
    mutual_info = 0
    for i, true_count in zip(unique_true, counts_true):
        for j, pred_count in zip(unique_pred, counts_pred):
            intersect_count = np.sum((y_true == i) & (y_pred == j))
            if intersect_count > 0:
                mutual_info += (intersect_count / total) * np.log(
                    (intersect_count * total) / (true_count * pred_count))
    h_true = -np.sum((counts_true / total) * np.log(counts_true / total))
    h_pred = -np.sum((counts_pred / total) * np.log(counts_pred / total))
    nmi = 2 * mutual_info / (h_true + h_pred) if (h_true + h_pred) > 0 else 0
    return nmi

# DB Index thủ công
def db_index_manual(X, clusters, centroids):
    k = len(centroids)
    intra_cluster_dists = np.zeros(k)
    for i in range(k):
        cluster_points = X[clusters == i]
        intra_cluster_dists[i] = np.mean([euclidean_distance(point, centroids[i]) for point in cluster_points])
    db_index = 0
    for i in range(k):
        max_ratio = max([(intra_cluster_dists[i] + intra_cluster_dists[j]) / euclidean_distance(centroids[i], centroids[j])
                         for j in range(k) if i != j], default=0)
        db_index += max_ratio
    return db_index / k if k > 1 else 0

# 10. Tính và in các chỉ số đánh giá
f1 = f1_score_manual(y_true, y_pred_mapped)
rand_index = rand_index_manual(y_true, y_pred_mapped)
nmi = nmi_manual(y_true, y_pred_mapped)
db_index = db_index_manual(X, y_pred_mapped, centroids)

print("Đánh giá chất lượng phân cụm:")
print("F1-score:", f1)
print("Rand Index:", rand_index)
print("NMI (Normalized Mutual Information):", nmi)
print("DB (Davies-Bouldin) Index:", db_index)
