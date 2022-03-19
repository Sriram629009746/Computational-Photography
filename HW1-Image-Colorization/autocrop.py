from utils import read_strip,circ_shift,find_shift,find_shift_pyramid,automatic_cropping
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # Setting the input output file path
    imageDir = '../Images/'
    outDir = '../Results/'
    
    default_crop = False
    
    
    jpg_files = []

    for file in os.listdir(imageDir):
        if file.endswith('.jpg'):
            jpg_files.append(file)

    for imageName in jpg_files:

        print("working on {}".format(imageName))
        print("--------------------------------")

        # Get r, g, b channels from image strip
        r, g, b = read_strip((imageDir + imageName))
        margin = 20
        
        if(default_crop == True):           
            b = b[margin:(b.shape[0]-margin),margin:(b.shape[1]-margin)]
            r = r[margin:(r.shape[0]-margin),margin:(r.shape[1]-margin)]
            g = g[margin:(g.shape[0]-margin),margin:(g.shape[1]-margin)]
            margin = 0
            
        r_to_shift, g_to_shift, b_to_shift = r,g,b
        r_shift, g_shift, b_shift = r,g,b
        
        
        r_blur = cv2.GaussianBlur(r,(3,3), 0)
        r_shift = cv2.Sobel(src=r_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

        g_blur = cv2.GaussianBlur(g,(3,3), 0)
        g_shift = cv2.Sobel(src=g_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

        b_blur = cv2.GaussianBlur(b,(3,3), 0)
        b_shift = cv2.Sobel(src=b_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

        # Calculate shift
        rShift = find_shift(r_shift, b_shift, margin)
        print("rshift: {}".format(rShift))
        gShift = find_shift(g_shift, b_shift, margin)
        print("gshift: {}".format(gShift))

        # Shifting the images using the obtained shift values
        finalB = b_to_shift
        finalG = circ_shift(g_to_shift, gShift)
        finalR = circ_shift(r_to_shift, rShift)
        
        left,right,top,bottom = automatic_cropping(finalR, finalG, finalB, 20)
        finalR = finalR[top:bottom,left:right]
        finalG = finalG[top:bottom,left:right]
        finalB = finalB[top:bottom,left:right]

        # Putting together the aligned channels to form the color image
        finalImage = np.stack((finalR, finalG, finalB), axis = 2)

        # Writing the image to the Results folder
        plt.imsave(outDir + imageName[:-4] + '_sobel_edges_crop.jpg', finalImage)
        
        
    tif_images = []

    for file in os.listdir(imageDir):
        if file.endswith('.tif'):
            tif_images.append(file)
            
    for imageName in tif_images:

        print("working on {}".format(imageName))
        print("--------------------------------")

        # Get r, g, b channels from image strip
        r, g, b = read_strip((imageDir + imageName))
        margin = 200

        if(default_crop == True):           
            b = b[margin:(b.shape[0]-margin),margin:(b.shape[1]-margin)]
            r = r[margin:(r.shape[0]-margin),margin:(r.shape[1]-margin)]
            g = g[margin:(g.shape[0]-margin),margin:(g.shape[1]-margin)]
            margin = 0

        r_to_shift, g_to_shift, b_to_shift = r,g,b
        r_shift, g_shift, b_shift = r,g,b

        r_blur = cv2.GaussianBlur(r,(3,3), 0)
        r_shift = cv2.Sobel(src=r_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

        g_blur = cv2.GaussianBlur(g,(3,3), 0)
        g_shift = cv2.Sobel(src=g_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

        b_blur = cv2.GaussianBlur(b,(3,3), 0)
        b_shift = cv2.Sobel(src=b_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

        # Calculate shift
        rShift = find_shift_pyramid(r_shift, b_shift, margin)
        print("rshift: {}".format(rShift))
        gShift = find_shift_pyramid(g_shift, b_shift, margin)
        print("gshift: {}".format(gShift))

        # Shifting the images using the obtained shift values
        finalB = b_to_shift
        finalG = circ_shift(g_to_shift, gShift)
        finalR = circ_shift(r_to_shift, rShift)
        
        left,right,top,bottom = automatic_cropping(finalR, finalG, finalB, 400)

        # Putting together the aligned channels to form the color image
        finalImage = np.stack((finalR, finalG, finalB), axis = 2)

        # Writing the image to the Results folder
        plt.imsave(outDir + imageName[:-4] + '_sobel_edge_crop.tiff', finalImage) #saving in tiff file as matplotlib save doesn't support tifs
        
        break