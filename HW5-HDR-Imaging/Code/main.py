""" Assignment 5 - Starter code


"""
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import cv2
from random import sample
from PIL import Image

# Based on code by James Tompkin
#
# reads in a directory and parses out the exposure values
# files should be named like: "xxx_yyy.jpg" where
# xxx / yyy is the exposure in seconds. 
def ParseFiles(calibSetName, dir):
    imageNames = os.listdir(os.path.join(dir, calibSetName))
    
    filePaths = []
    exposures = []
    
    for imageName in imageNames:
        exposure = imageName.split('.')[0].split('_')
        exposures.append(int(exposure[0]) / int(exposure[1]))
        filePaths.append(os.path.join(dir, calibSetName, imageName))
    
    # sort by exposure
    sortedIndices = np.argsort(exposures)[::-1]
    filePaths = [filePaths[i] for i in sortedIndices]
    exposures = [exposures[i] for i in sortedIndices]
    
    return filePaths, exposures

def Read(path):
    # Read image and return
    # return plt.imread(path)
    return np.asarray(Image.open(path))

def triange_window(x):
    if(x<=128):
        return x
    else:
        return 255 - x

def gsolve(Z, B, lambda_, w):

    Zmax = 255

    n = Zmax + 1
    num_px, num_im = Z.shape
    A = np.zeros((num_px * num_im + n, n + num_px))
    b = np.zeros((A.shape[0]))

    # include the data fitting equations
    k = 0
    for i in range(num_px):
        for j in range(num_im):
            #print(Z[i,j])
            wij = w[Z[i,j]]
            A[k, Z[i,j]] = wij
            A[k, n+i] = -wij
            b[k] = wij * B[j]
            k += 1

    # fix the curve by setting its middle value to 0
    A[k, n//2] = 1
    k += 1

    # include the smoothness equations
    for i in range(n-2):
        A[k, i]= lambda_ * w[i+1]
        A[k, i+1] = -2 * lambda_ * w[i+1]
        A[k, i+2] = lambda_ * w[i+1]
        k += 1

    # solve the system using LLS
    output = np.linalg.lstsq(A, b)
    x = output[0]
    g = x[:n]
    lE = x[n:]

    return [g, lE]


def plot_crf(crf_channel, Zmax):
    plt.figure(figsize=(24,8))
    channel_names = ['red', 'green', 'blue']
    for ch in range(3):
        plt.subplot(1,3,ch+1)
        plt.plot(crf_channel[ch], np.arange(Zmax+1), color=channel_names[ch], linewidth=2)
        plt.xlabel('log(X)')
        plt.ylabel('Pixel intensity')
        plt.title('CRF for {} channel'.format(channel_names[ch]))

    plt.figure(figsize=(8,8))
    for ch in range(3):
        plt.plot(crf_channel[ch], np.arange(Zmax+1), color=channel_names[ch], linewidth=2, label=channel_names[ch]+' channel')
    plt.xlabel('log(X)')
    #plt.xlim([-10, 10])
    plt.ylabel('Pixel intensity')
    plt.title('Camera Response Function'.format(channel_names[ch]))
  
    plt.legend()
    
def compute_irradiance(crf_ch, w, images, B):
    H,W,C = images[0].shape
    num_images = len(images)
  
    # irradiance map for each color channel
    irradiance_map = np.empty((H*W, 3))
    for ch in range(3):
        crf = crf_ch[ch]
        num_ = np.empty((num_images, H*W)) 
        den_ = np.empty((num_images, H*W))
        for j in range(num_images):
            flat_image = (images[j][:,:,ch].flatten()).astype(np.uint8)
            num_[j, :] = np.multiply(w[flat_image], crf[flat_image] - B[j])
            den_[j, :] = w[flat_image]

        irradiance_map[:, ch] = np.sum(num_, axis=0) / (np.sum(den_, axis=0) + 1e-6)

    irradiance_map = np.reshape(np.exp(irradiance_map), (H,W,C))
  
    return irradiance_map

def global_tonemap(ir_map, gamma):
    global_tonemapped_img = np.zeros_like(ir_map)
    # Perform both local and global tone-mapping

    for ch in range(3):
        global_tonemapped_img[:,:,ch] = ir_map[:,:,ch]/np.max(ir_map[:,:,ch])
        #global_tonemapped_img[:,:,ch] = ir_map[:,:,ch]/(1+ir_map[:,:,ch])
        global_tonemapped_img[:,:,ch] = np.power(global_tonemapped_img[:,:,ch], gamma)
        
    return global_tonemapped_img

def local_tonemap(ir_map, sigma=2.0, dR=4, gamma=0.1,  ):
    I = np.mean(ir_map, axis=2)
    chroma_R = ir_map[:,:,0]/I
    chroma_G = ir_map[:,:,1]/I
    chroma_B = ir_map[:,:,2]/I
    L = np.log2(I)
    #sigma = 2.0
    halfwidth = int(3*sigma)
    k = 2*halfwidth + 1
    B = cv2.GaussianBlur(L, (k, k),sigma)#, sigmaX=sigma)
    D = L-B
    #dR = 5
    s = dR/(np.max(B)-np.min(B))
    B1 = (B - np.max(B))*s
    O = 2**(B1+D)
    chroma_R = np.multiply(O,chroma_R)
    chroma_G = np.multiply(O,chroma_G)
    chroma_B = np.multiply(O,chroma_B)
    local_tonemap_img = np.stack([chroma_R,chroma_G,chroma_B],axis=2)
    local_result = local_tonemap_img**gamma
    local_result = np.clip(local_result,0,1.0)
    
    return local_result
    
	


# Setting up the input output paths and the parameters




inputDir = '../Images/'
outputDir = '../Results/'

#_lambda = 50


#calibSetName_list = ['Chapel', 'Office']
#calibSetName = 'Chapel'
calibSetName = 'Office'
# Parsing the input images to get the file names and corresponding exposure
# values
filePaths, exposures = ParseFiles(calibSetName, inputDir)


""" Task 1 """

# Sample the images
P = len(filePaths)
N = 5*256/(P-1)
N = int(N+1)
print(P,N)

img = plt.imread(filePaths[0])
rows = img.shape[0]
cols = img.shape[1]
images = []
Z = np.zeros((N,P,3))
k=0

random_pixel_indices = [random.randint(0, rows*cols) for p in range(N)]
for img_path in filePaths:
    
    img1 = Read(img_path)
    img = img1.copy()
    #img *= 255 # or any coefficient
    img = img.astype(np.uint8)
    images.append(img)
    img_r = np.array((img[:,:,0].reshape(1,-1))[0])
    Z[:,k,0] = (img_r[random_pixel_indices]).astype(np.uint8)
    img_g = np.array((img[:,:,1].reshape(1,-1))[0])
    Z[:,k,1] = (img_g[random_pixel_indices]).astype(np.uint8)
    img_b = np.array((img[:,:,2].reshape(1,-1))[0])
    Z[:,k,2] = (img_b[random_pixel_indices]).astype(np.uint8)
    k=k+1


# Recover the camera response function (CRF) using Debevec's optimization code (gsolve.m)
B = np.log(exposures)
w_args = np.arange(0,256)
w = np.array([triange_window(x) for x in w_args])

########################################################
#lambda_list = [50]#,30,40]#,30]#,40,50,60,70,80,90]
#gamma_list = [0.1]#,0.2,0.4]#,0.8,0.9]
#sigma_list = [1.0,2.0]
lambda_ = 50
########################################################## 

crf_ch = []
logE_ch = []


for ch in range(3):
    #print(Z[:,:,ch])
    crf, logE = gsolve(Z[:,:,ch].astype(np.uint8), B, lambda_, w)
    crf_ch.append(crf)
    logE_ch.append(logE)
    
plot_crf(crf_ch, 255)

# Reconstruct the radiance using the calculated CRF
ir_map = compute_irradiance(crf_ch, w, images, B)
global_tonemap_img = global_tonemap(ir_map, 0.15)
local_tonemapped_img = local_tonemap(ir_map, 1, 8, 0.3)

plt.imshow(global_tonemap_img)
plt.imshow(local_tonemapped_img)