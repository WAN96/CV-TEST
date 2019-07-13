import cv2
import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

lable_list= ['0','20','40','60','80','100','120','140','160']



img = cv2.imread('./000625_1.jpg')
img = cv2.resize(img, (64, 128))

x = cv2.Sobel(img, cv2.CV_16S, 1, 0, 1)
absX = cv2.convertScaleAbs(x)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1, 1)
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

col = dst.shape[0]
row = dst.shape[1]
ksize = 8

direct = np.zeros((dst.shape[0], dst.shape[1]))
magnitude = np.zeros_like(direct)
HOG = np.zeros((int(col / ksize), int(row / ksize), 9))

for i in range(int(col / ksize)):
    for j in range(int(row / ksize)):
        magnitude[i:i + 8, j:j + 8] = np.max((absX[i:i + 8, j:j + 8, :] * absX[i:i + 8, j:j + 8, :]
                                              + absY[i:i + 8,j:j + 8,:] * absY[i:i + 8,j:j + 8,:]) ** 0.5,
                                             axis=2)
        direct[i:i + 8, j:j + 8] = np.max(np.arctan(absY[i:i + 8, j:j + 8, :] / absX[i:i + 8, j:j + 8, :])*180/np.pi,
                                          axis=2)
        magnitudep = magnitude[i:i + 8, j:j + 8]
        directp = direct[i:i + 8, j:j + 8]
        for h in range(ksize):
            for k in range(ksize):
                if (directp[h, k] >= 0 and directp[h, k] < 20):
                    HOG[i, j, 0] = HOG[i, j, 0] + m.fabs(directp[h, k] - 20) / 20 * magnitudep[h, k]
                    HOG[i, j, 1] = HOG[i, j, 1] + m.fabs(directp[h, k] - 0) / 20 * magnitudep[h, k]
                if (directp[h, k] >= 20 and directp[h, k] < 40):
                    HOG[i, j, 1] = HOG[i, j, 1] + m.fabs(directp[h, k] - 40) / 20 * magnitudep[h, k]
                    HOG[i, j, 2] = HOG[i, j, 2] + m.fabs(directp[h, k] - 20) / 20 * magnitudep[h, k]
                if (directp[h, k] >= 40 and directp[h, k] < 60):
                    HOG[i, j, 2] = HOG[i, j, 2] + m.fabs(directp[h, k] - 60) / 20 * magnitudep[h, k]
                    HOG[i, j, 3] = HOG[i, j, 3] + m.fabs(directp[h, k] - 40) / 20 * magnitudep[h, k]
                if (directp[h, k] >= 60 and directp[h, k] < 80):
                    HOG[i, j, 3] = HOG[i, j, 3] + m.fabs(directp[h, k] - 80) / 20 * magnitudep[h, k]
                    HOG[i, j, 4] = HOG[i, j, 4] + m.fabs(directp[h, k] - 60) / 20 * magnitudep[h, k]
                if (directp[h, k] >= 80 and directp[h, k] < 100):
                    HOG[i, j, 4] = HOG[i, j, 4] + m.fabs(directp[h, k] - 100) / 20 * magnitudep[h, k]
                    HOG[i, j, 5] = HOG[i, j, 5] + m.fabs(directp[h, k] - 80) / 20 * magnitudep[h, k]
                if (directp[h, k] >= 100 and directp[h, k] < 120):
                    HOG[i, j, 5] = HOG[i, j, 5] + m.fabs(directp[h, k] - 120) / 20 * magnitudep[h, k]
                    HOG[i, j, 6] = HOG[i, j, 6] + m.fabs(directp[h, k] - 100) / 20 * magnitudep[h, k]
                if (directp[h, k] >= 120 and directp[h, k] < 140):
                    HOG[i, j, 6] = HOG[i, j, 6] + m.fabs(directp[h, k] - 140) / 20 * magnitudep[h, k]
                    HOG[i, j, 7] = HOG[i, j, 7] + m.fabs(directp[h, k] - 120) / 20 * magnitudep[h, k]
                if (directp[h, k] >= 140 and directp[h, k] < 160):
                    HOG[i, j, 7] = HOG[i, j, 7] + m.fabs(directp[h, k] - 160) / 20 * magnitudep[h, k]
                    HOG[i, j, 8] = HOG[i, j, 8] + m.fabs(directp[h, k] - 140) / 20 * magnitudep[h, k]
                if (directp[h, k] >= 160):
                    HOG[i, j, 8] = HOG[i, j, 8] + m.fabs(directp[h, k] - 180) / 20 * magnitudep[h, k]
                    HOG[i, j, 0] = HOG[i, j, 0] + m.fabs(directp[h, k] - 160) / 20 * magnitudep[h, k]
step = 1
normalsize = 2
print(HOG.shape)
#print("Before normalization:{}".format(HOG[0,0,:]))
finalHOG = np.zeros((int((HOG.shape[0]-normalsize)/step+1),int((HOG.shape[1]-normalsize)/step+1),36))

for ni in range(-1,HOG.shape[0]-2):
    for nj in range(-1, HOG.shape[1]-2):
      finalHOG[ni+1,nj+1] = np.reshape(HOG[ni+1:ni+1+normalsize, nj+1:nj+1+normalsize,:]/\
         np.sum(np.sqrt(HOG[ni+1:ni+1+normalsize, nj+1:nj+1+normalsize,:]
                        *HOG[ni+1:ni+1+normalsize, nj+1:nj+1+normalsize,:])),(36,))

#print("After normalizition:{}".format(HOG[0,0,:]))


plt.bar(x = range(9),height= HOG[0,0,:],color = 'blue', label = 'HOG')
plt.xticks([index for index in range(9)], lable_list)
plt.show()


finalHOG = finalHOG.reshape(-1,1)
print(finalHOG.shape)


for lx in range(int(col / ksize)):
    for lz in range(int(row / ksize)):
        cv2.line(img, (0,int(lx*8)),(int(row),int(lx*8)),color=(255,255,0), thickness=1)
        cv2.line(img, (int(lz*8),0),(int(lz*8),int(col)),color=(255,255,0), thickness=1)

cv2.imshow('origin', cv2.resize(img,(300,600)))
cv2.imshow('sobel ', cv2.resize(dst,(300,600)))

cv2.imwrite('./HOG1.jpg', dst)
cv2.imwrite('./HOG2.jpg', img)
cv2.waitKey(0)
