import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
img = cv2.imread('anhtoanha.jpg', 0)

# Bộ lọc Gaussian để giảm nhiễu
gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)

# Sobel
sobelx = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

# Prewitt
prewittx = cv2.filter2D(gaussian_blur, cv2.CV_32F, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
prewitty = cv2.filter2D(gaussian_blur, cv2.CV_32F, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
prewitt = cv2.magnitude(prewittx, prewitty)

# Robert
robertx = cv2.filter2D(gaussian_blur, cv2.CV_32F, np.array([[1, 0], [0, -1]]))
roberty = cv2.filter2D(gaussian_blur, cv2.CV_32F, np.array([[0, 1], [-1, 0]]))
robert = cv2.magnitude(robertx, roberty)

# Canny
canny = cv2.Canny(gaussian_blur, 50, 150)

# Hiển thị kết quả
plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(2, 3, 2), plt.imshow(gaussian_blur, cmap='gray'), plt.title('Gaussian Blur')
plt.subplot(2, 3, 3), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
plt.subplot(2, 3, 4), plt.imshow(prewitt, cmap='gray'), plt.title('Prewitt')
plt.subplot(2, 3, 5), plt.imshow(robert, cmap='gray'), plt.title('Robert')
plt.subplot(2, 3, 6), plt.imshow(canny, cmap='gray'), plt.title('Canny')
plt.show()
