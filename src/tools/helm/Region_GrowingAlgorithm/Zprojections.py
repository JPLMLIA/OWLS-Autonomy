# -*- coding: utf-8 -*-

"""
Created on  September 28 2022
@author: Max Riekeles, riekeles@tu-berin.de  
Version 1.0
"""

from skimage import io
import numpy as np
import os
import cv2 
from datetime import datetime

    #select folders (time points)
for folder in range (505 ,853):
    start = datetime.now()
    #directory
    dir = '' + str(folder) +'/'
    
    listfiles =[]
    featured_list =[]
    listfiles_without_ext=[]
    current_image =[]
    image_array = []
    glist_w = []
    glist_w2 = []
    glist_wTEST = []
    for img_file in os.listdir(dir):
        listfiles.append(int(float(os.path.splitext(img_file)[0])))
    
    listfiles.sort()
    #a = startvalue of plane, b = end value of plane, c = jump over every c-plane
    a = 100 
    b = 201 
    c = 1 
    images_sorted = [cv2.imread(dir + str(fileNr) + '.000.tif', cv2.IMREAD_GRAYSCALE) for fileNr in listfiles]
    out = np.zeros(images_sorted[0].shape, np.uint8) 
    
    def compare():
        for n in range(a,b,c):
           current_image = images_sorted[n]
           np.maximum(out, current_image, out)
        io.imshow(out) 
    compare()
    #write the images
    cv2.imwrite(str(a)+str(b)+str(c)+"REAL_max-z-projection_00" + str(folder) + ".tif", out)
    end = datetime.now()
    print(end-start, folder)
    
