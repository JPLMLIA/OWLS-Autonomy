# -*- coding: utf-8 -*-
"""
Created on  September 28 2022

@author: Max Riekeles, riekeles@tu-berin.de 
Big thanks to Hadi Albalkhi for helping greatly 
with the implementation! Helpful link for this
implementation was: https://stackoverflow.com/questions/43923648/region-growing-python

Version 1.0
"""



   
import cv2
import numpy as np
from numpy import savetxt
import ctypes
import pyautogui
clicks = []# Define radius of Region Growing and define threshold 
RADIUS = 9
THRESHOLD = 10 


# setting seeds
for f in range(100, 1024, 100):
    for g in range(100, 1024, 100):
        # defines seed point according tou mouseclick 
        def on_mouse(event, x, y, flags, params):  
             
             
             if 1 == 1:
                clicks.append([f, g])
   
        # get the coordinates of neighbour pixels for a given pixel in accordance to the given radius  
        def get_all_neighbours(pixel, radius, max_coordinates): 
            
            neighboursCoordinatesList = []
            currentX = pixel[0]
            currentY = pixel[1]
            y_max = max_coordinates[0]-1
            x_max = max_coordinates[1]-1
            
            
            # add all possible indices to the list neighboursIndices 
            for i in range(-radius, radius): 
                if i == 0:
                    continue
               
                neighboursCoordinatesList.append([currentX+i, currentY+i]);
                neighboursCoordinatesList.append([currentX+i, currentY]);
                neighboursCoordinatesList.append([currentX, currentY+i]);
        
            # now filter all invalid indices - those are  
            #  - indices with negative values (all arrays start with index 0 - no 
            #    negative indices) 
            #  - indices with absolute values which exceed the total length (minus one)
            #    of the array on the corresponding axis
            for neighboursCoordinates in neighboursCoordinatesList:
                neighboursX = neighboursCoordinates[0]
                neighboursY = neighboursCoordinates[1]
                if(neighboursX < 0 or neighboursY < 0 or neighboursX > x_max or neighboursY > y_max):
                    neighboursCoordinatesList.remove(neighboursCoordinates) 
              
            return neighboursCoordinatesList
        # Cost function: determine if a neighbour pixel is similar to a given pixel
        # load corresponding timestamps (for example npy file)->use as intensity image. 
        # Compare values of pixels. The differnece should not be greater than a given 
        # threshold 
        def is_in_region(current_pixel, neighbour, img, intensity_image):
            current_pixel_intensity = intensity_image[current_pixel[1], current_pixel[0]]
            neighbour_pixel_intensity = intensity_image[neighbour[1], neighbour[0]]
            # ignore background regions
            if current_pixel_intensity >770: 
                return False  
           
            vec_time_diff = current_pixel_intensity * 1. - neighbour_pixel_intensity * 1.
           
            if (abs(vec_time_diff) <= THRESHOLD):
                return True       
            else:
                return False
        # Implement Region Growing Algorithm: Starting with a given seed, the algorithm 
        # checks all the neighbour within a given radius and determine according to the 
        # cost function if the pixels belong to the region or not. For the pixels that
        # are assigned to the region, the algorithm will process them recursivley.
        def region_growing(img, intensity_image, seed, y_max, x_max):
            list_of_region_pixels = [seed]
            out_img = np.zeros_like(img)
            out_img[seed[1], seed[0]] = [255, 255, 255]
            processed_pixels_matrix = np.zeros((img.shape[0], img.shape[1]))
            processed_pixels_matrix[seed[1], seed[0]] = 1
            iterations = img.shape[0]*img.shape[1]
            while len(list_of_region_pixels):
                current_pixel = list_of_region_pixels[0]
                for neighbour in get_all_neighbours(current_pixel, RADIUS, img.shape):
                    if neighbour[1] > x_max or neighbour[0] > y_max or neighbour[1] < 0 or neighbour[0] < 0:
                        continue
                    if processed_pixels_matrix[neighbour[1], neighbour[0]] == 0:
                        processed_pixels_matrix[neighbour[1], neighbour[0]] = 1
                        if is_in_region(current_pixel, neighbour, img, intensity_image):
                            out_img[current_pixel[1], current_pixel[0]] = [255, 255, 255]
                            list_of_region_pixels.append(neighbour)
                
                list_of_region_pixels.pop(0)
                iterations -= 1 
            #arr_reshaped = np.reshape(out_img, (1024, 1024))
            arr_reshaped = out_img.reshape(out_img.shape[1], -1) 
            reshaped_arr_reshaped = cv2.resize(arr_reshaped, (1024, 1024))
            non_zero_reshaped = np.nonzero(reshaped_arr_reshaped)
            print(non_zero_reshaped)
            print(len(non_zero_reshaped[0]))
            # exclude too short tracks, save found tracks
            if len(non_zero_reshaped[0]) < 20: 
                print('no tracks found')
            else:
             savetxt('Track' + str(f) +'_' + str(g) + '.csv', arr_reshaped, delimiter=',')
            return out_img

        
        #simulation of mouseclicks and keyboardclicks
        MOUSE_LEFTDOWN = 0x0002     # left button down
        MOUSE_LEFTUP = 0x0004       # left button up
 
        def left_click():
           ctypes.windll.user32.mouse_event(MOUSE_LEFTDOWN)  # left down
           ctypes.windll.user32.mouse_event(MOUSE_LEFTUP)
        
        #keyboard = Controller()
        def space_click(): 

            pyautogui.press('space')
            pyautogui.keyUp('space')
            
            
        def main():
            #insert here path to npy file and image
            intensity_image = np.load("mhi.npy")
            img = cv2.imread('mhi.png', 1)
            
            y_max = 1024-1 # max_coordinates[0]-1
            x_max = 1024-1 # max_coordinates[1]-1
            cv2.namedWindow('Input')
            cv2.setMouseCallback('Input', on_mouse, 0, )
            cv2.imshow('Input', img)
            cv2.waitKey()
            seed = clicks[-1]
            cv2.imshow('Input', region_growing( img, intensity_image, seed, y_max, x_max))
            cv2.waitKey(500)
            cv2.imshow('Input', img)
            left_click()
            space_click()

            
        main()
import VisualizeTracks.py
VisualizeTracks
