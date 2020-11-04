# 2. Implement the fill contour using morphological operations algorithm presented during lecture 5.
import numpy as np
import cv2 as cv
import sys

raw_image = cv.imread(cv.samples.findFile("..\\resources\\contour_fill_raw_image.png"))
grayscale_image = cv.cvtColor(raw_image, cv.COLOR_BGR2GRAY)

# get the inverted binary image (i.e. the complement of the contour):
res, inverted_binary_image = cv.threshold(grayscale_image, type=cv.THRESH_BINARY_INV, maxval=255, thresh=0)

# now, apply the Flood Fill algorithm:

"""
Get the greatest value from the sub-matrix defined by the kernel and the source image
param source_image: a numpy matrix representing the image on which the kernel must be applied
param kernel: a numpy matrix representing the kernel that must be applied on the source image;
                the kernel must be a square matrix and the side must be odd.
param center_x: the abscissa of the point on which the anchor of the kernel must be superposed
param center_y: the ordinate of the point on which the anchor of the kernel must be superposed
"""
def apply_dilation_kernel(source_image, kernel, center_x, center_y):
    lim_x_1, lim_x_2 = center_x - int(kernel.shape[1] / 2), center_x + int(kernel.shape[1] / 2)
    lim_y_1, lim_y_2 = center_y - int(kernel.shape[0] / 2), center_y + int(kernel.shape[0] / 2)
    source_interesting_portion = source_image[lim_y_1:lim_y_2, lim_x_1:lim_x_2] * kernel
    return source_interesting_portion.max()

"""
    
"""
def flood_fill(source_image, start_point_y, start_point_x):
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    initial_image = np.array(source_image)

    # get the complement of the source_image:
    grayscale_image = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)
    res, inverted_binary_image = cv.threshold(grayscale_image, type=cv.THRESH_BINARY_INV, maxval=255, thresh=0)

    # do the flood fill:
    
    return initial_image
