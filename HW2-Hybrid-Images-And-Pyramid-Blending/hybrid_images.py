""" Assignment 2 - Starter code

Credit: Alyosha Efros
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage.transform as sktr
from datetime import datetime


#mode 0: both gray
#mode 1: color + gray
#mode 2: gray + color
#mode 3: both color
mode = 3


def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)


def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int)(np.abs(2 * r + 1 - R))
    cpad = (int)(np.abs(2 * c + 1 - C))
    return np.pad(
        im, [(0 if r > (R - 1) / 2 else rpad, 0 if r < (R - 1) / 2 else rpad),
             (0 if c > (C - 1) / 2 else cpad, 0 if c < (C - 1) / 2 else cpad),
             (0, 0)], 'constant')


def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy


def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    print("in align_image_centers p1: {} , p2:{}".format(p1, p2))
    print("in align_image_centers p1: {} , p2:{}".format(p3, p4))
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2


def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    len2 = np.sqrt((p4[1] - p3[1]) ** 2 + (p4[0] - p3[0]) ** 2)
    dscale = len2 / len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=2)
    else:
        im2 = sktr.rescale(im2, 1. / dscale, channel_axis=2)
    return im1, im2


def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta * 180 / np.pi)
    return im1, dtheta


def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2 - h1) / 2.)): -int(np.ceil((h2 - h1) / 2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1 - h2) / 2.)): -int(np.ceil((h1 - h2) / 2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2 - w1) / 2.)): -int(np.ceil((w2 - w1) / 2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1 - w2) / 2.)): -int(np.ceil((w1 - w2) / 2.)), :]
    print(im1.shape, im2.shape)
    assert im1.shape == im2.shape
    return im1, im2


def align_images(im1, im2):
    pts = get_points(im1, im2)
    print("pts: {}".format(pts))
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


def gaussian2d(filter_size=5, std_dev=1.0):
    x = np.linspace(-(filter_size - 1) / 2.0, (filter_size - 1) / 2.0, filter_size)
    gauss1d = np.exp(-0.5 * np.square(x) / np.square(std_dev))
    gauss2d = np.outer(gauss1d, gauss1d)
    return gauss2d / np.sum(gauss2d)


if __name__ == "__main__":

    imageDir = '../Images/'
    outDir = '../Results/'

    im1_name = 'brock.jpg'
    im2_name = 'tiger.jpg'

    # 1. load the images

    # Low frequency image
    im1 = plt.imread(imageDir + im1_name)  # read the input image
    info = np.iinfo(im1.dtype)  # get information about the image type (min max values)
    im1 = im1.astype(np.float32) / info.max  # normalize the image into range 0 and 1
    # im1 = im1[:, :, :3]# some images have 4 channels

    # High frequency image
    im2 = plt.imread(imageDir + im2_name)  # read the input image
    info = np.iinfo(im2.dtype)  # get information about the image type (min max values)
    im2 = im2.astype(np.float32) / info.max  # normalize the image into range 0 and 1
    # im2 = im2[:, :, :3]# some images have 4 channels

    # 2. align the two images by calling align_images
    im1_aligned, im2_aligned = align_images(im1, im2)
    if (mode==0): #both gray
        im1_aligned = np.mean(im1_aligned, axis=2)
        im2_aligned = np.mean(im2_aligned, axis=2)
        im1_lp = scipy.signal.convolve2d(im1_aligned, gaussian2d(7, 1.0), boundary='symm', mode='same')
        im2_hp = im2_aligned - scipy.signal.convolve2d(im2_aligned, gaussian2d(25, 7.0), boundary='symm', mode='same')

    elif(mode==1): #Lp color + hp gray
        im1_lp = np.ones_like(im1_aligned)
        for i in range(3):
            im1_lp[:, :, i] = scipy.signal.convolve2d(im1_aligned[:, :, i], gaussian2d(43, 7.0), boundary='symm',
                                                      mode='same')
        im2_aligned = np.mean(im2_aligned, axis=2)
        im2_hp = im2_aligned - scipy.signal.convolve2d(im2_aligned, gaussian2d(43, 7.0), boundary='symm', mode='same')
        im2_hp = np.stack([im2_hp, im2_hp, im2_hp], axis=2)

    elif(mode==2):#Lp gray + hp color
        im1_aligned = np.mean(im1_aligned, axis=2)
        im1_lp = scipy.signal.convolve2d(im1_aligned, gaussian2d(25, 7.0), boundary='symm', mode='same')
        im1_lp = np.stack([im1_lp, im1_lp, im1_lp], axis=2)

        im2_hp = np.ones_like(im2_aligned)
        for i in range(3):
            im2_hp[:, :, i] = im2_aligned[:, :, i] - scipy.signal.convolve2d(im2_aligned[:, :, i], gaussian2d(19, 3),
                                                                             boundary='symm', mode='same')
    elif(mode==3):#Lp color + hp color
        im1_lp = np.ones_like(im1_aligned)
        im2_hp = np.ones_like(im2_aligned)
        for i in range(3):
            im1_lp[:, :, i] = scipy.signal.convolve2d(im1_aligned[:, :, i], gaussian2d(43, 7.0), boundary='symm',
                                                      mode='same')
            im2_hp[:, :, i] = im2_aligned[:, :, i] - scipy.signal.convolve2d(im2_aligned[:, :, i], gaussian2d(19, 3.0),
                                                                             boundary='symm',
                                                                             mode='same')

    im_low = im1_lp
    im_high = im2_hp
    im = im_low + im_high

    im = im / im.max()
    im[np.where(im < 0)] = 0
    im[np.where(im > 1)] = 1

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_time = current_time.replace(':', '_')

    if (mode == 0):
        plt.imsave(outDir + im1_name[:-4] + '_' + im2_name[:-4] + '_hybrid_gray_' + current_time + '.jpg', im,
                   cmap='gray')
    elif (mode == 1):
        plt.imsave(outDir + im1_name[:-4] + '_' + im2_name[:-4] + '_hybrid_lp_color_hp_gray_' + current_time + '.jpg', im)
    elif (mode == 2):
        plt.imsave(outDir + im1_name[:-4] + '_' + im2_name[:-4] + '_hybrid_lp_gray_hp_color_' + current_time + '.jpg', im)
    elif (mode == 3):
        plt.imsave(outDir + im1_name[:-4] + '_' + im2_name[:-4] + '_hybrid_lp_color_hp_color_' + current_time + '.jpg', im)

    pass
