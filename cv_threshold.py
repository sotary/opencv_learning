#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 08:37:02 2018

@author: li
"""

import cv2   
from matplotlib import pyplot as plt
import tensorflow as tf    
import numpy as np 
def reversePic(src):  
        # 图像反转    
    for i in range(src.shape[0]):  
        for j in range(src.shape[1]):  
            src[i,j] = 255 - src[i,j]  
    return src   

im = cv2.imread("/home/li/anaconda_project/save_model_sample/e3.jpg")  
plt.subplot(321),plt.title('origin image'),plt.imshow(im)
im2 =reversePic(im)
plt.subplot(322),plt.title('reversed image'),plt.imshow(im2)
im3=cv2.threshold(im2,125,255,cv2.THRESH_TRUNC)[1]
plt.subplot(323),plt.title('thresed image'),plt.imshow(im3)
binary = cv2.threshold(im2,150,255,cv2.THRESH_BINARY)[1]
plt.subplot(324),plt.title('thres image'),plt.imshow(binary,'gray')
im4=im2[:,:,0]
binary2 = cv2.threshold(im2,150,255,cv2.THRESH_BINARY)[1]
plt.subplot(325),plt.title('thres image'),plt.imshow(binary2,'gray')