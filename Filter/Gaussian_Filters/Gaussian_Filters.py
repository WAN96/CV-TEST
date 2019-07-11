import cv2
import numpy as np
import math as m


def TwoDGaussian(sigma, x, y):
    G = m.exp(-(x ** 2 + y ** 2) / (2 * sigma * sigma)) / (2 * m.pi * sigma * sigma)
    return G


def Filter(sigma, kernelsize):
    f = np.zeros((kernelsize, kernelsize))
    center = (kernelsize - 1) / 2
    for i in range(kernelsize):
        for j in range(kernelsize):
            f[i, j] = TwoDGaussian(sigma, j - center, i - center)
    t = np.sum(f)
    f = f / t
    print(f)
    return f


img = cv2.imread('./003462.jpg')
im2 = np.zeros((img.shape[0], img.shape[1]))
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ksize = 5
F = Filter(1.5, ksize)
for i in range(img.shape[0] - ksize + 1):
    for j in range(img.shape[1] - ksize + 1):
        im2[i + 1, j + 1] = np.abs(np.sum(imgray[i:ksize + i, j:ksize + j] * F))

cv2.imshow('origin', img)
cv2.imshow('gray ', imgray)
cv2.imshow('filter', np.uint8(im2))
cv2.imwrite('./Gaussian_filter.jpg', im2)
cv2.waitKey(0)
