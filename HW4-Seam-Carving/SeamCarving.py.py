#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""" Assignment 4 - Starter code

""" 
# Import required libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from scipy.ndimage import convolve1d

# Read source and mask (if exists) for a given id
def Read(id, path = ""):
    source = plt.imread(path + "image_" + id + ".jpg") / 255
    maskPath = path + "mask_" + id + ".jpg"
    
    if os.path.isfile(maskPath):
        mask = plt.imread(maskPath)
        assert(mask.shape == source.shape), 'size of mask and image does not match'
        mask = (mask > 128)[:, :, 0].astype(int)
    else:
        mask = np.zeros_like(source)[:, :, 0].astype(int)

    return source, mask

def SeamCarve(input, color_img, widthFac, heightFac, mask, useMask):
    
    #input = np.array([[5,8,12,3],[4,2,3,9],[7,3,4,2],[6,5,7,8]])
    #print("input : {}".format(input))

    # Main seam carving function. This is done in three main parts: 1)
    # computing the energy function, 2) finding optimal seam, and 3) removing
    # the seam. The three parts are repeated until the desired size is reached.
    next_img_list = []
    E_list = []
    ret_list = []

    assert (widthFac == 1 or heightFac == 1), 'Changing both width and height is not supported!'
    assert (widthFac <= 1 and heightFac <= 1), 'Increasing the size is not supported!'
    
    inSize = input.shape
    retarget_size   = (int(heightFac*inSize[0]), int(widthFac*inSize[1]))
    
    img = input.copy()
    
    numSeams = inSize[1] - retarget_size[1]
    if(heightFac < 1):
        img = np.rot90(img,-1)
        #img = img.T
        color_img = np.rot90(color_img,-1)
        numSeams = inSize[0] - retarget_size[0]
        if(useMask==True):
            mask = np.rot90(mask,-1)
            #mask = mask.T

    retargeted_img = img.copy()
    color_img_1 = color_img.copy()
    #print(color_img_1.shape)
    #print(numSeams)

    for i in range(1, numSeams+1):
        
        #compute energy matrix
        E = retargeted_img.copy()
        ret_list.append(retargeted_img)
        
        next_img = np.zeros((img.shape[0], img.shape[1]-i))
        next_color_img = np.zeros((img.shape[0], img.shape[1]-i, 3))
        if(useMask== True):
            next_mask = np.zeros((img.shape[0], img.shape[1]-i))
        #print("next col")
        #print(next_color_img.shape,E.shape)
        
        x_edge_filter = np.array([[0,0,0],[-1,0,1],[0,0,0]])
        y_edge_filter = x_edge_filter.T
        
        #grad_x = abs(cv2.filter2D(E, ddepth=cv2.CV_64F, kernel=x_edge_filter))
        #grad_y = abs(cv2.filter2D(E, ddepth=cv2.CV_64F, kernel=y_edge_filter))
        
        #grad_x = abs(cv2.Sobel(E, ddepth=cv2.CV_64F, dx=1, dy=0))
        #grad_y = abs(cv2.Sobel(E, ddepth=cv2.CV_64F, dx=0, dy=1))
        weight = [1,-1]
        E = np.absolute(convolve1d(E, weights = weight, axis =0)) + np.absolute(convolve1d(E, weights = weight, axis =1))
        
        E_list.append(E)
        
        if(useMask==True):
            E+= mask
        
        Seam_track = np.zeros_like(E)
        #print(retargeted_img.shape)
        for i in range(1, retargeted_img.shape[0]):
            for j in range(retargeted_img.shape[1]):
                neigbors = []
                neigbors.append(E[i-1,j])
                neigbor_dict = {}
                neigbor_dict[0] = E[i-1,j]
                if(j!=0):
                    neigbors.append(E[i-1,j-1])
                    neigbor_dict[-1] = E[i-1,j-1]
                if(j<retargeted_img.shape[1]-1):
                    neigbors.append(E[i-1,j+1])
                    neigbor_dict[1] = E[i-1,j+1]
                    
                E[i,j] = E[i,j] + min(neigbors)
                Seam_track[i,j] = j + min(neigbor_dict,key=neigbor_dict.get)
        seam_low = np.argmin(E[-1])
        #print(Seam_track)
    
        
        for row_index in np.arange(retargeted_img.shape[0]-1,-1,-1):
                      
            row = retargeted_img[row_index]
            #print(row)
            row = np.delete(row,int(seam_low))
            next_img[row_index] = row.copy()
            #print("row_index:{}".format(row_index))
            
            for ch in range(3):
                #print("here")
                color_row_ch = color_img_1[row_index,:,ch]
                color_row_ch = np.delete(color_row_ch, int(seam_low))
                next_color_img[row_index,:,ch] = color_row_ch.copy()
            
            if(useMask==True):
                maskrow = mask[row_index]
                maskrow = np.delete(maskrow,int(seam_low))
                next_mask[row_index] = maskrow.copy()
                  
            seam_low = Seam_track[row_index,int(seam_low)]
            #print(seam_low)
            
        next_img_list.append(next_img)
        retargeted_img = next_img.copy()
        color_img_1 = next_color_img.copy()
        if(useMask==True):
            mask = next_mask.copy()
        #print(next_img)
        
    if(heightFac < 1):
        next_color_img = np.rot90(next_color_img)
        

    return next_color_img,retarget_size,ret_list, E_list, next_img_list
'''
def SeamCarve(input, color_img, widthFac, heightFac, mask, useMask):
    
    input = np.array([[5,8,12,3],[4,2,3,9],[7,3,4,2],[6,5,7,8]]).T
    heightFac = 0.5
    widthFac = 1
    useMask = False
    #print("input : {}".format(input))

    # Main seam carving function. This is done in three main parts: 1)
    # computing the energy function, 2) finding optimal seam, and 3) removing
    # the seam. The three parts are repeated until the desired size is reached.

    assert (widthFac == 1 or heightFac == 1), 'Changing both width and height is not supported!'
    assert (widthFac <= 1 and heightFac <= 1), 'Increasing the size is not supported!'
    
    inSize = input.shape
    retarget_size   = (int(heightFac*inSize[0]), int(widthFac*inSize[1]))
    
    img = input.copy()
    
    numSeams = inSize[1] - retarget_size[1]
    if(heightFac < 1):
        #img = np.rot90(img,-1)
        img = img.T
        #color_img = np.rot90(color_img,-1)
        numSeams = inSize[0] - retarget_size[0]
        if(useMask==True):
            #mask = np.rot90(mask,-1)
            mask = mask.T

    retargeted_img = img.copy()
    color_img_1 = color_img.copy()
    print(color_img_1.shape)
    print(numSeams)

    for i in range(1, numSeams+1):
        
        #compute energy matrix
        E = retargeted_img.copy()
        next_img = np.zeros((img.shape[0], img.shape[1]-i))
        next_color_img = np.zeros((img.shape[0], img.shape[1]-i, 3))
        if(useMask== True):
            next_mask = np.zeros((img.shape[0], img.shape[1]-i))
        #print("next col")
        #print(next_color_img.shape,E.shape)
        #
        x_edge_filter = np.array([[0,0,0],[-1,0,1],[0,0,0]])
        y_edge_filter = x_edge_filter.T
        
        #grad_x = abs(cv2.filter2D(E, ddepth=cv2.CV_64F, kernel=x_edge_filter))
        #grad_y = abs(cv2.filter2D(E, ddepth=cv2.CV_64F, kernel=y_edge_filter))
        
        #grad_x = abs(cv2.Sobel(E, ddepth=cv2.CV_64F, dx=1, dy=0))
        #grad_y = abs(cv2.Sobel(E, ddepth=cv2.CV_64F, dx=0, dy=1))
        
        #E = grad_x + grad_y
        #E = 
        if(useMask==True):
            E+= mask
        
        Seam_track = np.zeros_like(E)
        #print("out")
        for i in range(1, retargeted_img.shape[0]):
            for j in range(retargeted_img.shape[1]):
                neigbors = [E[i-1,j]]
                neigbor_dict = {}
                neigbor_dict[0] = E[i-1,j]
                if(j!=0):
                    neigbors.append(E[i-1,j-1])
                    neigbor_dict[-1] = E[i-1,j-1]
                if(j<retargeted_img.shape[1]-1):
                    neigbors.append(E[i-1,j+1])
                    neigbor_dict[1] = E[i-1,j+1]
                    
                E[i,j] = E[i,j] + min(neigbors)
                Seam_track[i,j] = j + min(neigbor_dict,key=neigbor_dict.get)
 
        seam_low = np.argmin(E[-1])
        
        for row_index in np.arange(retargeted_img.shape[0]-1,-1,-1):
                      
            row = retargeted_img[row_index]
            #print(row)
            row = np.delete(row,int(seam_low))
            next_img[row_index] = row.copy()
            #print("row_index:{}".format(row_index))
            
            for ch in range(3):
                color_row_ch = color_img_1[row_index,:,ch]
                color_row_ch = np.delete(color_row_ch, int(seam_low))
                next_color_img[row_index,:,ch] = color_row_ch.copy()
            
            if(useMask==True):
                maskrow = mask[row_index]
                maskrow = np.delete(maskrow,int(seam_low))
                next_mask[row_index] = maskrow.copy()
                  
            seam_low = Seam_track[row_index,int(seam_low)]
            
        
        retargeted_img = next_img.copy()
        #color_img_1 = next_color_img.copy()
        if(useMask==True):
            mask = next_mask.copy()
        print(next_img)
        
    if(heightFac < 1):
        #next_color_img = np.rot90(next_color_img)
        next_img = next_img.T
        
    print(next_img)
    print("done")

    return next_img,retarget_size#next_color_img
'''


# Setting up the input output paths
inputDir = '../Images/'
outputDir = '../Results/'

widthFac = 0.5; # To reduce the width, set this parameter to a value less than 1
heightFac = 1;  # To reduce the height, set this parameter to a value less than 1
N = 10 # number of images

for index in range(1, N + 1):
    if(index!=10):
        continue
    print(index)
    
    widthFac = 0.5; # To reduce the width, set this parameter to a value less than 1
    heightFac = 1;  # To reduce the height, set this parameter to a value less than 1
    if((index==3) or(index==4) or (index==5)):
        widthFac = 1 
        heightFac = 0.5
    input, mask = Read(str(index).zfill(2), inputDir)
    if(np.max(input)==255):
        input = input/255.0
    if(index == 4):
        useMask = True
    
    grayimg = np.mean(input, axis=2)
    # Performing seam carving. This is the part that you have to implement.
    output, size,ret_list, E_list, nex_list  = SeamCarve(grayimg, input, widthFac, heightFac, mask, useMask=False)
    #SeamCarve(input, widthFac, heightFac, mask)
    #break
    
    # Writing the result
    
    plt.imsave("{}/resultx_{}_{}x{}.jpg".format(outputDir, 
                                            str(index).zfill(2), 
                                            str(size[0]).zfill(2), 
                                            str(size[1]).zfill(2)), output)
    print("done")
    #break
    

