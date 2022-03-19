""" Assignment 3 - Starter code


""" 

# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Read source, target and mask for a given id
def Read(id, path = ""):
    source = plt.imread(path + "source_" + id + ".jpg")
    info = np.iinfo(source.dtype) # get information about the image type (min max values)
    source = source.astype(np.float32) / info.max # normalize the image into range 0 and 1
    target = plt.imread(path + "target_" + id + ".jpg")
    info = np.iinfo(target.dtype) # get information about the image type (min max values)
    target = target.astype(np.float32) / info.max # normalize the image into range 0 and 1
    mask   = plt.imread(path + "mask_" + id + ".jpg")
    info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    mask = mask.astype(np.float32) / info.max # normalize the image into range 0 and 1

    return source, mask, target

# Adjust parameters, source and mask for negative offsets or out of bounds of offsets
def AlignImages(mask, source, target, offset):
    sourceHeight, sourceWidth, _ = source.shape
    targetHeight, targetWidth, _ = target.shape
    xOffset, yOffset = offset

    if (xOffset < 0):
        mask    = mask[abs(xOffset):, :]
        source  = source[abs(xOffset):, :]
        sourceHeight -= abs(xOffset)
        xOffset = 0
    if (yOffset < 0):
        mask    = mask[:, abs(yOffset):]
        source  = source[:, abs(yOffset):]
        sourceWidth -= abs(yOffset)
        yOffset = 0
    # Source image outside target image after applying offset
    if (targetHeight < (sourceHeight + xOffset)):
        sourceHeight = targetHeight - xOffset
        mask    = mask[:sourceHeight, :]
        source  = source[:sourceHeight, :]
    if (targetWidth < (sourceWidth + yOffset)):
        sourceWidth = targetWidth - yOffset
        mask    = mask[:, :sourceWidth]
        source  = source[:, :sourceWidth]

    maskLocal = np.zeros_like(target)
    maskLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = mask
    sourceLocal = np.zeros_like(target)
    sourceLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = source

    return sourceLocal, maskLocal

def max_val(a,b):
    if(abs(a)>abs(b)):
        return a
    else:
        return b

def PoissonBlend(source, mask, target, isMix):

    mask_rgb = mask
    blended_img = np.ones_like(target)
    n_cnt = 4
    a_vals = [-1,-1,n_cnt,-1,-1]

    a_list = []
    b_list = []
    x_list = []

    for ch in range(3):
        row_list = []
        col_list = []
        data_list = []

        s = source[:,:,ch]
        t = target[:,:,ch]
        mask = mask_rgb[:,:,ch]

        k = t.shape[0] * t.shape[1]
        m = t.shape[0]
        n = t.shape[1]
        b = np.ones(k)

        for col in range(mask.shape[1]):
            row = 0
            for x in mask[:,col]:

                if(x == 1):
                    index = m*col+ row
                    neighbors = []
                    a_vals = []
                    n_cnt = 0
                    s_left = 0
                    s_right = 0
                    s_up = 0
                    s_down = 0
                    t_left = 0
                    t_right = 0
                    t_up = 0
                    t_down = 0
                    grad_left = 0
                    grad_right = 0
                    grad_up = 0
                    grad_down = 0

                    if(col>0):
                        left  = m*(col-1) + row
                        neighbors.append(left)
                        n_cnt+=1
                        s_left = s[row,col-1]
                        t_left = t[row,col-1]
                        a_vals.append(-1)
                        if(isMix):
                            grad_left = max_val((s[row,col] - s_left), (t[row,col] - t_left))


                    if(row>0):
                        above = m*col+ (row-1)
                        neighbors.append(above)
                        n_cnt+=1
                        s_up = s[row-1,col]
                        t_up = t[row-1,col]
                        #a_vals[1] = -1
                        a_vals.append(-1)
                        if(isMix):
                            grad_up = max_val((s[row,col] - s_up), (t[row,col] - t_up))


                    if(row<(m-1)):
                        below = m*col+ (row+1)
                        neighbors.append(below)
                        n_cnt+=1
                        s_down = s[row+1,col]
                        t_down = t[row+1,col]
                        a_vals.append(-1)
                        if(isMix):
                            grad_down = max_val((s[row,col] - s_down), (t[row,col] - t_down))

                    if(col<(n-1)):
                        right = m*(col+1) + row
                        neighbors.append(right)
                        n_cnt+=1
                        s_right = s[row,col+1]
                        t_right = t[row,col+1]
                        a_vals.append(-1)
                        if(isMix):
                            grad_right = max_val((s[row,col] - s_right), (t[row,col] - t_right))


                    neighbors.append(index)
                    a_vals.append(n_cnt)

                    for iter_, num in enumerate(neighbors):
                        row_list.append(index)
                        data_list.append(a_vals[iter_])
                        col_list.append(num)

                    if(isMix==True):
                        '''
                        b[index] = max_val((s[row,col] - s_left), (t[row,col] - t_left)) +\
                        max_val((s[row,col] - s_right), (t[row,col] - t_right)) +\
                        max_val((s[row,col] - s_up), (t[row,col] - t_up)) +\
                        max_val((s[row,col] - s_down), (t[row,col] - t_down))
                        '''
                        b[index] = grad_left + grad_up + grad_down + grad_right
                    else:
                        #n_cnt = 4
                        b[index] = n_cnt*s[row,col] - s_left - s_right - s_up - s_down

                else:
                    index = m*col+ row
                    row_list.append(index)
                    data_list.append(1)
                    col_list.append(index)
                    b[index] = t[row,col]
                row+=1


        r = np.array(row_list)
        c = np.array(col_list)
        d = np.array(data_list)
        a_new = csr_matrix((d, (r, c)), shape=(k, k))
        b_new = csr_matrix(b)

        soln = spsolve(a_new,b_new.reshape(k,-1))
        res =  soln.reshape(-1,1).reshape(t.shape[1],t.shape[0]).T
        a_list.append(a_new)
        b_list.append(b_new)
        x_list.append(soln)
        res = np.clip(res, 0.0, 1.0)

        blended_img[:,:,ch] = res

    return blended_img #, x_list, a_list, b_list


if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/new/'

    # False for source gradient, true for mixing gradients
    isMix = True

    # Source offsets in target
    offsets = [[210, 10], [10, 28], [140, 80], [-40, 90], [60, 100], [20, 20],
               [-28, 88]]  # , [50,50],[0,0], [150,100], [150,100]]

    # main area to specify files and display blended image
    for index in range(len(offsets)):

        # Read data and clean mask
        source, maskOriginal, target = Read(str(index + 1).zfill(2), inputDir)

        # Cleaning up the mask
        mask = np.ones_like(maskOriginal)
        mask[maskOriginal < 0.5] = 0

        # Align the source and mask using the provided offest
        source, mask = AlignImages(mask, source, target, offsets[index])

        ### The main part of the code ###

        # Implement the PoissonBlend function
        poissonOutput = PoissonBlend(source, mask, target, isMix)

        # Writing the result

        if not isMix:
            plt.imsave("{}poisson_{}.jpg".format(outputDir, str(index + 1).zfill(2)), poissonOutput)
        else:
            plt.imsave("{}poisson_{}_Mixing.jpg".format(outputDir, str(index + 1).zfill(2)), poissonOutput)
        # break
