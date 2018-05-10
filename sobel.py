#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 08:39:41 2018

@author: li
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/li/图片/object1.png',0)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)#默认ksize=3
sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1)
laplacian = cv2.Laplacian(img,cv2.CV_64F)#默认ksize=3
#人工生成一个高斯核，去和函数生成的比較
kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],np.float32)#
img1 = np.float64(img)#转化为浮点型的
img_filter = cv2.filter2D(img1,-1,kernel)
sobelxy1 = cv2.Sobel(img1,-1,1,1)
img_filter2 = cv2.filter2D(img_filter,-1,kernel)

plt.subplot(221),plt.imshow(sobelx,'gray')
plt.subplot(222),plt.imshow(sobely,'gray')
plt.subplot(223),plt.imshow(sobelxy,'gray')
plt.subplot(224),plt.imshow(laplacian,'gray')

plt.figure()
plt.subplot(131),plt.imshow(img,'gray')
plt.subplot(132),plt.imshow(img_filter,'gray')
plt.subplot(133),plt.imshow(img_filter2,'gray')