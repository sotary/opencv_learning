# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 08:35:47 2018

@author: Administrator
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import datetime
img_rgb = cv2.imread("C:/Users/Administrator/AnacondaProjects/learning_opencv/img/load_pic1.jpg")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
plt.subplot(321),plt.imshow(img_rgb,label='原图'),plt.title('origin')
template = img_gray[550:560,500:510]
w, h = template.shape[::-1]
#开始计时
begin = datetime.datetime.now()
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCORR_NORMED)
threshold = 0.995

end=datetime.datetime.now()
total_time=end-begin
print(total_time)
print(res.shape)
#umpy.where(condition[, x, y])
#Return elements, either from x or y, depending on condition.
#If only condition is given, return condition.nonzero().

loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
#结束计时

#plt.imshow(img_rgb)  
#先把匹配上的画成绿框

#cv2.imwrite('res4.png',img_rgb) 
#我们对中间区域进行涂色 
img_contours=img_rgb[250:520,0:800]
gray=cv2.cvtColor(img_contours,cv2.COLOR_BGR2GRAY)
plt.subplot(322),plt.imshow(gray)
plt.subplot(323),plt.imshow(img_contours)
plt.subplot(324),plt.imshow(gray,'gray')


#找矩形，先阈值分割一下，二值化，再把找到的矩形放在contours里
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

contours,hierarchy,tt = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img_contours,hierarchy,-1,(0,0,255),11)
#plt.imshow(img_contours)
plt.subplot(325),plt.imshow(img_contours)
img_contours_gray2=cv2.cvtColor(img_contours,cv2.COLOR_BGR2GRAY)
rest2,binary = cv2.threshold(img_contours_gray2,100,255,cv2.THRESH_BINARY)
plt.subplot(326),plt.imshow(img_contours_gray2,'gray')
#plt.subplot(122),plt.imshow(img_contours_gray2)

#cnt = contours[0]
#M = cv2.moments(cnt)
#print( M)


