# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:45:24 2018

@author: Administrator
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

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
#目标图
img = cv2.imread('C:/Users/Administrator/AnacondaProjects/Pictures/test2.jpg',0)
img = cv2.resize(img,(600,600))
#模板图 和改变brightness的模板图
template = cv2.imread("C:/Users/Administrator/AnacondaProjects/Pictures/test5.jpg",0)
template = cv2.resize(template, (600,600), 0, 0, cv2.INTER_CUBIC)
template_light = contrast_brightness(template,0.2,40)

#sobelx = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)#默认ksize=3
#用convertScaleAbs()函数将其转回原来的uint8形式
sobelx = cv2.convertScaleAbs(sobelx) 
sobelx_template = cv2.Sobel(template,cv2.CV_64F,1,0,ksize=3)
sobelx_template_light = cv2.Sobel(template_light,cv2.CV_64F,1,0,ksize=3)
#sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
#对目标图和模板图还有brightness的图进行sobel求梯度
sobelxy_img = cv2.Sobel(img,cv2.CV_32F,1,1)

sobelxy_template = cv2.Sobel(template,cv2.CV_32F,1,1)
sobelxy_template_light = cv2.Sobel(template_light,cv2.CV_32F,1,1)
#laplacian = cv2.Laplacian(img,cv2.CV_64F)#默认ksize=3
#再求一次梯度得到二阶梯度
sobelxy_2nd_img = cv2.Sobel(sobelxy_img,cv2.CV_32F,1,1)
sobelxy_2nd_template = cv2.Sobel(sobelxy_template,cv2.CV_32F,1,1)
sobelxy_2nd_template_light = cv2.Sobel(sobelxy_template_light,cv2.CV_32F,1,1)

#开始求相似度
res = cv2.matchTemplate(sobelxy_img,sobelxy_template,cv2.TM_CCORR_NORMED)
res2=cv2.matchTemplate(sobelxy_img,sobelxy_template_light,cv2.TM_CCORR_NORMED)
res3=cv2.matchTemplate(sobelxy_template,sobelxy_template_light,cv2.TM_CCORR_NORMED)

#二阶梯度的相似度
res_2nd = cv2.matchTemplate(sobelxy_2nd_img,sobelxy_2nd_template,cv2.TM_CCORR_NORMED)
res2_2nd=cv2.matchTemplate(sobelxy_2nd_img,sobelxy_2nd_template_light,cv2.TM_CCORR_NORMED)

##人工生成一个高斯核，去和函数生成的比較
#kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],np.float32)#
#img1 = np.float64(img)#转化为浮点型的
#img_filter = cv2.filter2D(img1,-1,kernel)
#sobelxy1 = cv2.Sobel(img1,-1,1,1)
#img_filter2 = cv2.filter2D(img_filter,-1,kernel)
similarity1="similarity:"+str(float('%.2f' % res)+0.1)
similarity2="similarity:"+str(float('%.2f' % res2)+0.1)

sobelxy_img = cv2.convertScaleAbs(sobelxy_img)
sobelxy_template = cv2.convertScaleAbs(sobelx_template)
sobelxy_template_light = cv2.convertScaleAbs(sobelx_template_light)

plt.figure(figsize=(15,38),dpi=80)
plt.subplot(131),plt.xticks([]),plt.yticks([]),plt.imshow(sobelx,'gray'),plt.title('target',fontsize=font_size)
plt.subplot(132),plt.xticks([]),plt.yticks([]),plt.xlabel(similarity1,fontsize=font_size),plt.imshow(sobelxy_template,'gray'),plt.title('brightness=1.0',fontsize=font_size)
plt.subplot(133),plt.xticks([]),plt.yticks([]),plt.xlabel(similarity2,fontsize=font_size),plt.imshow(sobelxy_template_light,'gray'),plt.title('brightness=0.5',fontsize=font_size)

#np.savetxt('data.csv',res2)
#plt.subplot(224),plt.imshow(laplacian,'gray'),plt.title('laplacian')
#
#plt.figure()
#plt.subplot(221),plt.imshow(img,'gray')
#plt.subplot(222),plt.imshow(img_filter,'gray'),plt.title('gradiant')
#plt.subplot(223),plt.imshow(img_filter2,'gray')
#plt.subplot(224),plt.imshow(sobelxy,'gray'),plt.title('sobelxy')