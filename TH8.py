# Import các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# 1. Tải dữ liệu CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. Tiền xử lý dữ liệu
# Chuẩn hóa dữ liệu (từ 0-255 xuống [0, 1])
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Chuyển đổi hình ảnh thành vector 1 chiều (flatten)
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# 3. Chạy mô hình KNN
print("----- Running KNN -----")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_flat, y_train.ravel())

# Dự đoán và đánh giá
y_pred_knn = knn.predict(x_test_flat)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# 4. Chạy mô hình SVM
print("\n----- Running SVM -----")
svm = SVC(kernel='linear')
svm.fit(x_train_flat, y_train.ravel())

# Dự đoán và đánh giá
y_pred_svm = svm.predict(x_test_flat)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# 5. Chạy mô hình ANN
print("\n----- Running ANN -----")
model = Sequential()
model.add(Dense(128, input_dim=x_train_flat.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 lớp cho CIFAR-10

# Biên dịch mô hình
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(x_train_flat, y_train, epochs=10, batch_size=64)

# Dự đoán và đánh giá
y_pred_ann = model.predict(x_test_flat)
y_pred_ann = np.argmax(y_pred_ann, axis=1)  # Chuyển đổi kết quả softmax thành nhãn
print("ANN Accuracy:", accuracy_score(y_test, y_pred_ann))

# 6. Hiển thị kết quả so sánh
print("\n----- Final Results -----")
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn)}")
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print(f"ANN Accuracy: {accuracy_score(y_test, y_pred_ann)}")
