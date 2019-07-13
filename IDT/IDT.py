import numpy as np
import cv2

img = cv2.imread('./000217.jpg')

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x = cv2.Sobel(imgray, cv2.CV_16S, 1, 0)
absX = cv2.convertScaleAbs(x)
y = cv2.Sobel(imgray, cv2.CV_16S, 0, 1)
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

E = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
r = 0.98
a = 1 / 3

mask1 = np.array([[1, 1, 1], [1, 0, 1]])
mask2 = np.array([[1, 0, 1], [1, 1, 1]])
for i in range(1, dst.shape[0]):
    for j in range(1, dst.shape[1] - 1):
        E[i, j] = np.max(np.array([np.max(E[i - 1:i + 1, j - 1:j + 2] * mask1) * r, E[i, j]]))

for i in range(dst.shape[0] - 2, -1, -1):
    for j in range(dst.shape[1] - 2, 0, -1):
        E[i, j] = np.max(np.array([np.max(E[i:i + 2, j - 1:j + 2] * mask2) * r, E[i, j]]))

cv2.imshow('IDT1', np.uint8(E))

D = (1 - a) * E + a * dst

cv2.imshow('IDT', np.uint8(D))
cv2.imwrite("./IDT.jpg",np.uint8(D))
cv2.waitKey(0)
