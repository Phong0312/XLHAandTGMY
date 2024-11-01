import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier\
from sklearn.metrics import classification_report, accuracy_score

# Đọc dữ liệu từ file CSV
data = pd.read_csv("duong_dan_den_file_cua_ban.csv")  # Thay đường dẫn đến file CSV của bạn
# Xác định các biến đặc trưng và nhãn
X = data.drop("column_nhan", axis=1).values  # Thay 'column_nhan' bằng tên cột nhãn trong dữ liệu của bạn
y = data["column_nhan"].values  # Cột nhãn phân loại

# Các tỷ lệ chia tập train-test
ratios = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.4, 0.6)]

# Lưu kết quả
results = {}

# Thực hiện phân lớp với từng tỷ lệ
for train_ratio, test_ratio in ratios:
    ratio_label = f"{int(train_ratio * 100)}-{int(test_ratio * 100)}"
    results[ratio_label] = {}

    print(f"\nTrain-Test Split: {ratio_label}")

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, test_size=test_ratio,
                                                        random_state=42)

    # Huấn luyện mô hình SVM
    model_svm = svm.SVC(kernel='linear', random_state=42)
    model_svm.fit(X_train, y_train)
    y_pred_svm = model_svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    results[ratio_label]['SVM'] = {
        "accuracy": accuracy_svm,
        "classification_report": classification_report(y_test, y_pred_svm, output_dict=True)
    }
    print(f"SVM Accuracy ({ratio_label}): {accuracy_svm}")
    print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

    # Huấn luyện mô hình KNN
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    results[ratio_label]['KNN'] = {
        "accuracy": accuracy_knn,
        "classification_report": classification_report(y_test, y_pred_knn, output_dict=True)
    }
    print(f"KNN Accuracy ({ratio_label}): {accuracy_knn}")
    print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

# Kết quả tổng hợp
print("\n=== Tổng hợp Kết quả ===")
for ratio, result in results.items():
    print(f"\nTỷ lệ Train-Test {ratio}:")
    for model, metrics in result.items():
        print(f"\nModel: {model}")
        print(f"Accuracy: {metrics['accuracy']}")
        print("Classification Report:", metrics['classification_report'])
