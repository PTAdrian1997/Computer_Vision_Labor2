# 2. Implement the fill contour using morphological operations algorithm presented during lecture 5.
import numpy as np
import cv2 as cv
import sys
import queue

raw_image = cv.imread(cv.samples.findFile("..\\resources\\contour_fill_raw_image.png"))
grayscale_image = cv.cvtColor(raw_image, cv.COLOR_BGR2GRAY)

# get the inverted binary image (i.e. the complement of the contour):
res1, binary_image = cv.threshold(grayscale_image, type=cv.THRESH_BINARY, maxval=255, thresh=0)
res2, inverted_binary_image = cv.threshold(grayscale_image, type=cv.THRESH_BINARY_INV, maxval=255, thresh=0)

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
    print(lim_y_1, " ", lim_y_2, " ", lim_x_1, " ", lim_x_2, "\n")
    source_interesting_portion = source_image[lim_y_1:lim_y_2+1, lim_x_1:lim_x_2+1] * kernel
    return source_interesting_portion.min()

"""
:param source_image: numpy matrix
:param kernel: numpy matrix
"""
def dilate(source_image, kernel):
    result = np.array(source_image)
    kernel_side = kernel.shape[0]
    image_height, image_width = source_image.shape[0], source_image.shape[1]
    start_x, end_x = int(kernel_side/2), image_width - int(kernel_side/2)
    start_y, end_y = int(kernel_side/2), image_height - int(kernel_side/2)
    for current_y in range(start_y, end_y):
        for current_x in range(start_x, end_x):
            result[current_y, current_x] = apply_dilation_kernel(source_image, kernel, current_x, current_y)
    return result

"""
"""
def dilate_recursive(source_image, kernel, start_point_y, start_point_x):

    kernel_side = kernel.shape[0]
    image_height, image_width = source_image.shape[0], source_image.shape[1]
    x_min_lim, x_max_lim = int(kernel_side/2), image_width - int(kernel_side/2)
    y_min_lim, y_max_lim = int(kernel_side/2), image_height - int(kernel_side/2)

    result_image = np.array(source_image)
    q1 = queue.Queue(maxsize=0)
    q1.put((start_point_y, start_point_x))
    neighbouring_vectors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while q1.qsize() != 0:
        y, x = q1.get()
        old_value = result_image[y, x]
        result_image[y, x] = apply_dilation_kernel(source_image, kernel, x, y)
        if result_image[y, x] != old_value or (y == start_point_y and x == start_point_x):
            # push all the 0 neighbouring nodes in the queue:
                for vector_y, vector_x in neighbouring_vectors:
                    new_y, new_x = y + vector_y, x + vector_x
                    if new_y >= y_min_lim and new_y < y_max_lim and x_min_lim <= new_x < x_max_lim \
                            and result_image[new_y, new_x] == source_image[start_point_y, start_point_x]:
                        q1.put((new_y, new_x))

    return result_image



"""
Intersection of two different images with the same shape. Both images must be binary.
param image_1: numpy matrix
param image_2: numpy matrix
param val_1: the value that must be considered 1
param val_2: the value of 0
"""
def intersect(image_1, image_2, val_1, val_2):
    result = np.array(image_1)

    return result

"""
"""
def flood_fill(source_image, start_point_y, start_point_x):
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    initial_image = np.array(source_image)

    # get the complement of the source_image:
    grayscale_image = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)
    res, inverted_binary_image = cv.threshold(grayscale_image, type=cv.THRESH_BINARY_INV, maxval=255, thresh=0)

    # do the flood fill:
    while(True):
        new_image = np.array(initial_image)
        new_image = intersect(dilate_recursive(new_image, kernel, start_point_y, start_point_x), inverted_binary_image)


    return initial_image

kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
print(kernel.shape)
dilated_image = dilate_recursive(binary_image, kernel, 232, 101)
cv.imshow("dilated image", dilated_image)
cv.imshow("binary image", binary_image)
cv.waitKey(0)
