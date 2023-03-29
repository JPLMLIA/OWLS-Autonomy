# -*- coding: utf-8 -*-
"""
Created on  September 28 2022

@author: Max Riekeles, riekeles@tu-berin.de 
Version 1.0
"""

#This code creates pngs out of the saved tracks.
from numpy import genfromtxt
from PIL import Image
import matplotlib
import numpy as np
import cv2
from matplotlib import cm
#This must be like setting seeds of the Tracking code
for f in range(100, 1024, 100):
    for g in range(100, 1024, 100):
        try:
            my_data = genfromtxt('Track' + str(f) +'_' + str(g) + '.csv', delimiter=',')
        except IOError:
           my_data = 1
           
        if np.all(my_data != 1):
            imS = cv2.resize(my_data, (1024, 1024))
            matplotlib.image.imsave('Track'  + str(f) + str(g) + '.png', imS, cmap = cm.gray)
