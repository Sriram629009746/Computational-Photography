""" Assignment 1 - Starter code


""" 

# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.filters import sobel_v, sobel_h
from skimage.filters import gaussian
import os
import cv2
import argparse


# Function to retrieve r, g, b planes from Prokudin-Gorskii glass plate images
def read_strip(path):
    image = plt.imread(path) # read the input image
    info = np.iinfo(image.dtype) # get information about the image type (min max values)
    image = image.astype(float) / info.max # normalize the image into range 0 and 1
    print(image.shape)

    height = int(image.shape[0] / 3)

    # For images with different bit depth
    scalingFactor = 255 if (np.max(image) <= 255) else 65535
    
    # Separating the glass image into R, G, and B channels
    b = image[: height, :]
    g = image[height: 2 * height, :]
    r = image[2 * height: 3 * height, :]
    
    #b = b[margin:(b.shape[0]-margin),margin:(b.shape[1]-margin)]
    #r = r[margin:(r.shape[0]-margin),margin:(r.shape[1]-margin)]
    #g = g[margin:(g.shape[0]-margin),margin:(g.shape[1]-margin)]
    return r, g, b

# circshift implementation similar to matlab
def circ_shift(channel, shift):
    shifted = np.roll(channel, shift[0], axis = 0) # along rows
    shifted = np.roll(shifted, shift[1], axis = 1) # along cols
    return shifted

# find_shift implementation for single-scale method
def find_shift(im1, im2, margin = 20):
    
    shifts = np.arange(-20,21)
    height = im1.shape[0]
    weight = im1.shape[1]
    
    #print("margin: {}".format(margin))
    
    im1 = im1[margin:(im1.shape[0]-margin),margin:(im1.shape[1]-margin)]
    im2 = im2[margin:(im2.shape[0]-margin),margin:(im2.shape[1]-margin)]
    
    count = 0
    tx = 0
    ty = 0
    for x_shift in shifts:
        for y_shift in shifts:
            shifted_im1 = circ_shift(im1,[x_shift,y_shift])
            squared_diff = np.square(shifted_im1 - im2)
            ssd = np.sum(squared_diff)
            
            if(count==0):
                min_ssd = ssd
                tx = x_shift
                ty = y_shift
            count+=1
            
            if(ssd < min_ssd):
                min_ssd = ssd
                tx = x_shift
                ty = y_shift
                
    return [tx, ty]


# find_shift implementation for multi-scale pyramid method
def find_shift_pyramid(im1, im2, margin=200):
    
    scales = [8,4,2,1]
    initial_shift = np.array([0,0])
    print("margin: {}".format(margin))
    #print("im1 shape : {}".format(im1.shape))
    #print("im2 shape : {}".format(im2.shape))
    
    im1 = im1[margin:(im1.shape[0]-margin),margin:(im1.shape[1]-margin)]
    im2 = im2[margin:(im2.shape[0]-margin),margin:(im2.shape[1]-margin)]
    
    for scale in scales:
        print("scale: {}".format(scale))
        #print("initial shift: {}".format(initial_shift))
        initial_shift = initial_shift * 2
        #print("initial shift_next: {}".format(initial_shift))
        im1_resized = resize(im1, (im1.shape[0] // scale, im1.shape[1] // scale),
                           anti_aliasing=True)
        im2_resized = resize(im2, (im2.shape[0] // scale, im2.shape[1] // scale),
                           anti_aliasing=True)
        
        #print("resized im1: {}".format(im1_resized.shape))
        #print("resized im2: {}".format(im2_resized.shape))

        shifts = np.arange(-20,21)
        height = im1_resized.shape[0]
        weight = im1_resized.shape[1]

        tx = 0
        ty = 0
        count = 0
        for x_shift in shifts:
            for y_shift in shifts:
                shifted_im1 = circ_shift(im1_resized,[(x_shift + initial_shift[0]),(y_shift + initial_shift[1])])
                squared_diff = np.square(shifted_im1 - im2_resized)
                ssd = np.sum(squared_diff)
                if(count==0):
                    min_ssd = ssd
                    tx = initial_shift[0] + x_shift
                    ty = initial_shift[1] + y_shift
                count+=1
                
                if(ssd < min_ssd):
                    min_ssd = ssd
                    tx = initial_shift[0] + x_shift
                    ty = initial_shift[1] + y_shift
                    
        initial_shift = np.array([tx, ty])
    
    return [tx,ty]
    
def automatic_cropping(r_channel, g_channel, b_channel,border_range=400):
    
    # cropping
    l_max = 0
    r_min = 10000
    t_max = 0
    b_min = 10000

    threshold = 4000
    for aligned_channel in [r_channel,g_channel,b_channel]:

        sobelx = cv2.Sobel(src=aligned_channel, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        sobely = cv2.Sobel(src=aligned_channel, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis

        m = sobelx.shape[0]
        n = sobelx.shape[1]
        
        expected_border_x = border_range
        expected_border_y = border_range

        #find left and right border
        suml=[]
        for i in range(0,n):
            suml.append(sobelx[:,i].sum())

        #plt.plot(range(0,n),(np.abs(suml)))

        suml=np.array(np.abs(suml))
        half = int(n/2)
        #locs = np.where(suml>4000)

        left_start = 0
        right_end  = n

        if (len((np.where(suml[0:expected_border_x]>threshold))[0]) > 0):
            left_start = (np.where(suml[0:expected_border_x]>threshold))[0][-1]
        if (len((np.where(suml[(n-expected_border_x):n]>threshold))[0]) > 0):
            right_end = (np.where(suml[(n-expected_border_x):n]>threshold))[0][0] + n - expected_border_x

        #find top and bottom border
        suml=[]
        for i in range(0,m):
            suml.append(sobely[i,:].sum())

        #plt.plot(range(0,m),(np.abs(suml)))

        suml=np.array(np.abs(suml))
        half = int(m/2)
        #locs = np.where(suml>4000)

        top_start = 0
        bottom_end = m
        #top and bottom borders
        if ((len((np.where(suml[0:expected_border_y]>threshold))[0])) > 0):
            top_start = (np.where(suml[0:expected_border_y]>threshold))[0][-1]
        if ((len((np.where(suml[(m-expected_border_y):m]>threshold))[0])) > 0):
            bottom_end = (np.where(suml[(m-expected_border_y):m]>threshold))[0][0] + m - expected_border_y
        
        
        #plt.imshow(aligned_channel[top_start:bottom_end ,left_start:right_end],cmap='gray')
        l_max = max(l_max,left_start)
        r_min = min(r_min,right_end)
        t_max = max(t_max,top_start)
        b_min = min(b_min,bottom_end)

        #print(l_max,r_min,t_max,b_min)
        
    return l_max,r_min,t_max,b_min