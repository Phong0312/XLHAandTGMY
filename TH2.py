import cv2
import numpy as np
import matplotlib.pyplot as plt


# Hàm hiển thị ảnh
def show_image(title, image):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# Đọc ảnh đầu vào
image = cv2.imread('download.jpg', cv2.IMREAD_GRAYSCALE)  # Đọc ảnh ở chế độ grayscale (ảnh xám)


# 1. Dò biên bằng toán tử Sobel
def sobel_edge_detection(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient theo hướng x
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient theo hướng y
    sobel_combined = cv2.magnitude(sobelx, sobely)  # Kết hợp gradient của x và y
    sobel_combined = np.uint8(sobel_combined)
    return sobel_combined


sobel_edges = sobel_edge_detection(image)
show_image("Sobel Edge Detection", sobel_edges)


# 2. Dò biên bằng Laplace of Gaussian (LOG)
def laplacian_of_gaussian(image):
    # Bước 1: Làm mượt ảnh bằng Gaussian
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

    # Bước 2: Dò biên bằng Laplace
    log_image = cv2.Laplacian(blurred_image, cv2.CV_64F, ksize=3)
    log_image = np.uint8(np.absolute(log_image))  # Lấy giá trị tuyệt đối của Laplacian
    return log_image


log_edges = laplacian_of_gaussian(image)
show_image("Laplace of Gaussian (LOG) Edge Detection", log_edges)
