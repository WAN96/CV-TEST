import cv2
import numpy as np
import math as m


def ValueGaussian(sigma, x):
    G = np.exp(-(x ** 2) / (2 * sigma * sigma)) / (m.sqrt(2 * m.pi) * sigma)
    return G


def SpaceGaussian(sigma, x, y):
    G = m.exp(-(x ** 2 + y ** 2) / (2 * sigma * sigma)) / (2 * m.pi * sigma * sigma)
    return G


def Bilateral_Filter(sigma1, sigma2, kernelsize, im):
    f = np.zeros((kernelsize, kernelsize))
    center = (kernelsize - 1) / 2
    for i in range(kernelsize):
        for j in range(kernelsize):
            f[i, j] = SpaceGaussian(sigma1, j - center, i - center)

    x = np.ones_like(im)
    x = x * im[int(center), int(center)]
    x = np.abs(im - x)
    f1 = ValueGaussian(sigma2, x)
    f = f * f1
    t = np.sum(f)
    f = f / t
    return f


img = cv2.imread('./003462.jpg')
im2 = np.zeros((img.shape[0], img.shape[1]))
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ksize = 3
print("Running!")
for i in range(img.shape[0] - ksize + 1):
    for j in range(img.shape[1] - ksize + 1):
        crop = imgray[i:ksize + i, j:ksize + j]
        F = Bilateral_Filter(1.5, 1, ksize, crop)
        im2[i + 1, j + 1] = np.abs(np.sum(imgray[i:ksize + i, j:ksize + j] * F))

cv2.imshow('origin', img)
cv2.imshow('gray ', imgray)
cv2.imshow('filter', np.uint8(im2))
cv2.imwrite('./Bilateral_filter.jpg', im2)
cv2.waitKey(0)
