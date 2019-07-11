import cv2
import numpy as np


img = cv2.imread('./003462.jpg')
img2 = cv2.bilateralFilter(img,5,20,50)

Fx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Fy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
img3 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgx = np.zeros((img3.shape[0],img3.shape[1]))
imgy = np.zeros((img3.shape[0],img3.shape[1]))
imgf = np.zeros((img3.shape[0],img3.shape[1]))
print(img3.shape)

print(img3.shape)
for i in range(img3.shape[0]-2):
    for j in range(img3.shape[1]-2):
        imgx[i+1,j+1]= np.abs(np.sum(img3[i:3+i,j:3+j]*Fx))
        imgy[i+1,j+1]= np.abs(np.sum(img3[i:3+i,j:3+j]*Fy))
        imgf[i+1,j+1]= (imgx[i+1,j+1]*imgx[i+1,j+1]+imgy[i+1,j+1]*imgy[i+1,j+1])**0.5

x = cv2.Sobel(img3,cv2.CV_16S,1,0)
absX = cv2.convertScaleAbs(x)
y = cv2.Sobel(img3,cv2.CV_16S,0,1)
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX,0.5,absY,0.5,0)

cv2.imshow('origin',np.uint8(imgf))
cv2.imshow('opencv ',dst)
cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
cv2.imwrite('./Sobel_filter.jpg',dst)
cv2.waitKey(0)

