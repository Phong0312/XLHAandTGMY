import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm hiển thị ảnh
def show_images(images, titles):
    plt.figure(figsize=(10,10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Đọc ảnh
image = cv2.imread('"C:\Users\phong\Downloads\z5853363342574_e994d48abf807af21d84d8e740d47bdd.jpg"')  # Thay thế đường dẫn ảnh

# 1. Ánh âm tính (Negative Image)
negative_image = cv2.bitwise_not(image)

# 2. Tăng độ tương phản (Contrast Stretching)
min_val = np.min(image)
max_val = np.max(image)
contrast_stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# 3. Biến đổi log (Log Transformation)
c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(1 + image.astype(np.float64)))
log_image = np.array(log_image, dtype=np.uint8)

# 4. Cân bằng Histogram (Histogram Equalization)
# Áp dụng cho ảnh kênh màu YCrCb
ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
hist_eq_image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

# Hiển thị ảnh sau khi xử lý
images = [image, negative_image, contrast_stretched, log_image, hist_eq_image]
titles = ['Original', 'Negative', 'Contrast Stretched', 'Log Transformed', 'Histogram Equalized']

show_images(images, titles)
