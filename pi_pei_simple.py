#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 11:02:26 2018

@author: li
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#from pylab import mpl
#
#mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
#mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
#改变对比度的函数
def contrast_brightness(src1, a, b):
    rows,cols=src1.shape
    src2 = np.zeros([rows ,cols], src1.dtype) 
    for i in range(rows):
        for j in range(cols):
            color=src1[i,j]*a+b
            if color>255:
                src2[i,j]=255
            elif color<0:
                src2[i,j]=0
            else :
                src2[i,j]=color
                
    return src2
#fontsize
font_size=17
img_rgb = cv2.imread("C:/Users/Administrator/AnacondaProjects/Pictures/test2.jpg")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img_gray = cv2.resize(img_gray, (600,600), 0, 0, cv2.INTER_CUBIC)

#设置figure大小
plt.figure(figsize=(15,38),dpi=80)
#plt.subplot(131),plt.imshow(img_rgb,label='原图'),plt.title('origin')
plt.subplot(131),plt.xticks([]),plt.yticks([]),plt.imshow(img_gray,label='h',cmap ='gray'),plt.title(u'target',fontsize=font_size)

template = cv2.imread("C:/Users/Administrator/AnacondaProjects/Pictures/test5.jpg",0)
template = cv2.resize(template, (600,600), 0, 0, cv2.INTER_CUBIC)
template_light = contrast_brightness(template,0.2,40)
#计算相似度
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCORR_NORMED)
res2=cv2.matchTemplate(img_gray,template_light,cv2.TM_CCORR_NORMED)
res3=cv2.matchTemplate(template,template_light,cv2.TM_CCORR_NORMED)
similarity1="similarity:"+str(float('%.2f' % res))
similarity2="similarity:"+str(float('%.2f' % res3))
plt.subplot(132),plt.xticks([]),plt.yticks([]),plt.xlabel(similarity1,fontsize=font_size),plt.imshow(template,label='t',cmap ='gray'),plt.title(u'brightness=1.0',fontsize=font_size)
plt.subplot(133),plt.xticks([]),plt.yticks([]),plt.xlabel(similarity2,fontsize=font_size),plt.imshow(template_light,label='t',cmap ='gray'),plt.title(u'brightness=0.5',fontsize=font_size)
#plt.subplot(133),plt.xticks([]),plt.yticks([]),plt.imshow(img_gray,label='h',cmap ='gray'),plt.title(u'target',fontsize=15)