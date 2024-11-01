import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Tải bộ dữ liệu Iris
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Phân chia dữ liệu Iris thành tập huấn luyện và kiểm tra
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_iris, y_train_iris)
y_pred_nb = nb_model.predict(X_test_iris)

print("Naive Bayes Accuracy on Iris:", accuracy_score(y_test_iris, y_pred_nb))
print(classification_report(y_test_iris, y_pred_nb))

# CART (Gini Index)
cart_model = DecisionTreeClassifier(criterion='gini')
cart_model.fit(X_train_iris, y_train_iris)
y_pred_cart = cart_model.predict(X_test_iris)

print("CART Accuracy on Iris:", accuracy_score(y_test_iris, y_pred_cart))
print(classification_report(y_test_iris, y_pred_cart))

# ID3 (Information Gain)
id3_model = DecisionTreeClassifier(criterion='entropy')
id3_model.fit(X_train_iris, y_train_iris)
y_pred_id3 = id3_model.predict(X_test_iris)

print("ID3 Accuracy on Iris:", accuracy_score(y_test_iris, y_pred_id3))
print(classification_report(y_test_iris, y_pred_id3))

# -------------------------- Xử lý Bộ Dữ liệu Ảnh Nha Khoa --------------------------
# Đường dẫn đến bộ dữ liệu ảnh nha khoa
dental_image_path = "nhakhoa/Panoramic radiographs with periapical lesions Dataset/Periapical Dataset/Periapical Lesions"

# Sử dụng ImageDataGenerator để tiền xử lý ảnh
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Tạo các batch cho tập huấn luyện và tập kiểm tra
train_generator = datagen.flow_from_directory(
    dental_image_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

test_generator = datagen.flow_from_directory(
    dental_image_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Tạo mô hình mạng nơ-ron
num_classes = len(train_generator.class_indices)
nn_model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Biên dịch mô hình
nn_model.compile(optimizer=Adam(),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Huấn luyện mô hình
nn_model.fit(train_generator, epochs=10, validation_data=test_generator)

# Đánh giá mô hình mạng nơ-ron trên tập kiểm tra
nn_loss, nn_accuracy = nn_model.evaluate(test_generator)
print("Neural Network Accuracy on Dental Images:", nn_accuracy)
