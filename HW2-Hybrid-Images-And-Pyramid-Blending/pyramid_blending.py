""" Assignment 2 - Starter code


"""

# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage.transform import resize
import scipy
from math import *
from GetMask import GetMask

# Read source, target and mask for a given id
def Read(id, path=""):
    source = plt.imread(path + "source_" + id + ".jpg")
    info = np.iinfo(source.dtype)  # get information about the image type (min max values)
    source = source.astype(np.float32) / info.max  # normalize the image into range 0 and 1
    target = plt.imread(path + "target_" + id + ".jpg")
    info = np.iinfo(target.dtype)  # get information about the image type (min max values)
    target = target.astype(np.float32) / info.max  # normalize the image into range 0 and 1
    mask = plt.imread(path + "mask_" + id + ".jpg")
    info = np.iinfo(mask.dtype)  # get information about the image type (min max values)
    mask = mask.astype(np.float32) / info.max  # normalize the image into range 0 and 1

    return source, mask, target

def pyrDown(img):
    m, n, ch = img.shape
    result = resize(img, (m / 2, n / 2, ch), anti_aliasing=True)
    return result

def pyrUp(img):
    m, n, ch = img.shape
    result = resize(img, (2 * m, 2 * n, ch), anti_aliasing=True)
    return result

def create_gaussian_pyramid(img, num_layers):
    current_layer = img.copy()
    layers = [current_layer]

    for i in range(1, num_layers):
        #current_layer = cv2.pyrDown(current_layer)
        current_layer = pyrDown(current_layer)
        layers.append(current_layer)

    return layers


def create_laplacian_pyramid(gauss_pyramid, num_layers):

    orange_copy = gauss_pyramid[num_layers-1]
    lp_orange = [orange_copy]
    for i in range(num_layers-1, 0, -1):
        #gaussian_expanded = cv2.pyrUp(gauss_pyramid[i])
        gaussian_expanded = pyrUp(gauss_pyramid[i])
        laplacian = cv2.subtract(gauss_pyramid[i - 1], gaussian_expanded)
        lp_orange.append(laplacian)

    return lp_orange


def add_borders(source, side):
    new_image = 0.5 * np.ones((side, side, 3))

    border1 = (side - source.shape[0]) / 2
    border1 = int(border1)

    border2 = (side - source.shape[1]) / 2
    border2 = int(border2)
    new_image[border1:(side - border1), border2:(side - border2), :] = source

    return new_image


def remove_borders(source, side):
    #new_image = np.zeros((side, side, 3))
    border1 = (source.shape[0] - side) / 2
    border1 = int(border1)

    border2 = (source.shape[1] - side) / 2
    border2 = int(border2)
    new_image = source[border1:(source.shape[0] - border1), border2:(source.shape[1] - border2), :]

    return new_image


def blend(laplacian_pyr_1, laplacian_pyr_2, mask_pyr):
    LS = []
    for l1, l2, mask in zip(laplacian_pyr_1, laplacian_pyr_2, mask_pyr):
        final = l1 * mask + l2 * (1.0 - mask)
        LS.append(final)
    return LS


# Pyramid Blend
def PyramidBlend(source, mask, target):
    m = np.max([source.shape[0], source.shape[1], target.shape[0], target.shape[1]])
    m = floor(log2(m)) + 1
    m = pow(m)

    source = add_borders(source, m)
    target = add_borders(target, m)
    mask = add_borders(mask, m)

    num_layers = 6
    gauss_src_pyr = create_gaussian_pyramid(source, num_layers)
    laplacian_src_pyr = create_laplacian_pyramid(gauss_src_pyr, num_layers)

    gauss_target_pyr = create_gaussian_pyramid(target, num_layers)
    laplacian_target_pyr = create_laplacian_pyramid(gauss_target_pyr, num_layers)

    gauss_mask_pyr = create_gaussian_pyramid(mask, num_layers)

    # merge_and_collapse_pyramid
    blended_pyramid = blend(laplacian_src_pyr, laplacian_target_pyr, reversed(gauss_mask_pyr))

    final_pyramid = [blended_pyramid[0]]
    current_layer = blended_pyramid[0]
    for i in range(1, len(blended_pyramid)):
        current_layer = np.add(blended_pyramid[i], pyrUp(current_layer))
        final_pyramid.append(current_layer)

    return final_pyramid
    # return gauss_src_pyr,gauss_target_pyr,gauss_mask_pyr , laplacian_src_pyr, laplacian_target_pyr
    # return source * mask + target * (1 - mask)


if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    # main area to specify files and display blended image

    index = 1

    # Read data and clean mask
    source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

    orig_shape = source.shape

    # Cleaning up the mask
    mask = np.ones_like(maskOriginal)
    mask[maskOriginal < 0.5] = 0


    # Implement the PyramidBlend function (Task 2)
    final_pyramid = PyramidBlend(source, mask, target)
    reconstructed_image = final_pyramid[-1]
    reconstructed_image = remove_borders(reconstructed_image, orig_shape[0])

    reconstructed_image[reconstructed_image < 0] = 0
    reconstructed_image[reconstructed_image > 1] = 1

    # Writing the result

    plt.imsave("{}pyramidnew_{}.jpg".format(outputDir, str(index).zfill(2)), reconstructed_image)
