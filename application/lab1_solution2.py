# 2. Implement the fill contour using morphological operations algorithm presented during lecture 5.
import numpy as np
import cv2 as cv
import time

raw_image = cv.imread(cv.samples.findFile("..\\resources\\contour_fill_input_image.png"))

# now, apply the Flood Fill algorithm:

"""
Get the greatest value from the sub-matrix defined by the kernel and the source image
param source_image: a numpy matrix representing the image on which the kernel must be applied
param kernel: a numpy matrix representing the kernel that must be applied on the source image;
                the kernel must be a square matrix and the side must be odd.
                also, the kernel must only contain 1 and 0; 1 values are placed on the interesting positions;
param center_x: the abscissa of the point on which the anchor of the kernel must be superposed
param center_y: the ordinate of the point on which the anchor of the kernel must be superposed
param not_indexes: 
"""
def apply_dilation_kernel(source_image, kernel, center_x, center_y, not_indexes):
    image_copy = np.array(source_image)
    lim_x_1, lim_x_2 = center_x - int(kernel.shape[1] / 2), center_x + int(kernel.shape[1] / 2)
    lim_y_1, lim_y_2 = center_y - int(kernel.shape[0] / 2), center_y + int(kernel.shape[0] / 2)
    source_interesting_portion = image_copy[lim_y_1:lim_y_2+1, lim_x_1:lim_x_2+1]
    source_interesting_portion[tuple(np.array(not_indexes).T)] = 255
    return source_interesting_portion.min()

"""
Intersection of two different images with the same shape. Both images must be binary.
param image_1: numpy matrix
param image_2: numpy matrix
param val_1: the value that must be considered 1 (i.e. the value that must be in both images
                at the same pixel in order to be preserved)
param val_2: the background value
"""
def intersect(image_1, image_2, val_1, val_2):
    indexes_1_y, indexes_1_x = np.where(image_1 == val_1)
    indexes_2_y, indexes_2_x = np.where(image_2 == val_1)
    indexes_1 = np.array(list(zip(indexes_1_y, indexes_1_x)))
    indexes_2 = np.array(list(zip(indexes_2_y, indexes_2_x)))
    binary_1 = np.zeros(image_1.shape)
    binary_1[tuple(indexes_1.T)] = 1
    binary_2 = np.zeros(image_2.shape)
    binary_2[tuple(indexes_2.T)] = 1
    intersection = np.logical_and(binary_1, binary_2)
    indexes_3_y, indexes_3_x = np.where(intersection == True)
    indexes_3 = list(zip(indexes_3_y, indexes_3_x))
    intersection = np.ones(image_1.shape) * val_2
    # intersection[[*np.array(indexes_3).T]] = val_1 # deprecated
    intersection[tuple(np.array(indexes_3).T)] = val_1
    return intersection


"""
Provide the result of applying a dilation with the provided kernel on the provided numpy matrix
param image_matrix: a numpy matrix representing a binary image
param kernel: a numpy matrix representing the dilation kernel
"""
def dilate(image_matrix, kernel):
    kernel_side = kernel.shape[0]
    result = np.array(image_matrix)
    x_min, x_max = 0, image_matrix.shape[1]
    y_min, y_max = 0, image_matrix.shape[0]

    # compute the list of unimportant coordinates:
    not_kernel = np.logical_not(kernel)
    not_indexes_y, not_indexes_x = np.where(not_kernel == 1)
    not_indexes = list(zip(not_indexes_y, not_indexes_x))

    for current_x in range(x_min, x_max):
        if current_x - int(kernel_side / 2) >= 0 and current_x + int(kernel_side / 2) < x_max:
            for current_y in range(y_min, y_max):
                if current_y - int(kernel_side / 2) >= 0 and current_y + int(kernel_side / 2) < y_max:
                    result[current_y, current_x] = apply_dilation_kernel(image_matrix, kernel, current_x, current_y,
                                                                         not_indexes)
    return result


"""
Provide the union of the two matrices; Basically a logical or, where 1 is replaced by val_1 and 
0 is replaced by val_2;
param image_matrix_1: a numpy matrix representing a binary image
param image_matrix_2: same as above
param val_1: the integer that must be treated as 1 in logical_or
param val_2: the integer that must be treated as 0 in logical_or
"""
def union(image_matrix_1, image_matrix_2, val_1, val_2):
    indexes_1_y, indexes_1_x = np.where(image_matrix_1 == val_1)
    indexes_2_y, indexes_2_x = np.where(image_matrix_2 == val_1)
    indexes_1 = np.array(list(zip(indexes_1_y, indexes_1_x)))
    indexes_2 = np.array(list(zip(indexes_2_y, indexes_2_x)))
    binary_1 = np.zeros(image_matrix_1.shape)
    binary_1[tuple(indexes_1.T)] = 1
    binary_2 = np.zeros(image_matrix_2.shape)
    binary_2[tuple(indexes_2.T)] = 1
    union = np.logical_or(binary_1, binary_2)
    indexes_3_y, indexes_3_x = np.where(union == True)
    indexes_3 = list(zip(indexes_3_y, indexes_3_x))
    union = np.ones(image_matrix_1.shape) * val_2
    union[tuple(np.array(indexes_3).T)] = val_1
    return union



"""
Fill the contour from source_image with content (i.e. pixels)
param source_image: a numpy matrix, representing an image that contains a closed line (i.e. the contour)
                    to be filled
param start_point_y: an integer representing the ordinate of the starting point
param start_point_x: an integer representing the abscissa of the starting point
"""
def flood_fill(source_image, start_point_y, start_point_x):
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    # kernel = np.array([[255, 0, 255], [0, 0, 0], [255, 0, 255]], dtype=np.uint8)
    # kernel = np.array([[0, 255, 0], [255, 255, 255], [0, 255, 0]], dtype=np.uint8)


    # get the complement of the source_image:
    grayscale_image = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)
    res, inverted_binary_image = cv.threshold(grayscale_image, type=cv.THRESH_BINARY_INV, maxval=255, thresh=0)
    res2, binary_image = cv.threshold(grayscale_image, type=cv.THRESH_BINARY, maxval=255, thresh=0)

    # binary_aux = np.array(binary_image)
    binary_aux = np.ones(binary_image.shape) * 255
    binary_aux[start_point_y, start_point_x] = 0

    # do the flood fill:
    i = 0
    while True:
        cv.imwrite("..\\output\\binary_aux_image_" + str(i) + ".png", binary_aux)
        #cv.imshow(" ", binary_aux)
        new_image = np.array(binary_aux, dtype=np.uint8)
        # dilated_image = dilate(new_image, kernel)
        dilated_image = cv.erode(new_image, kernel=kernel)
        # intersected_image = intersect(dilated_image, inverted_binary_image, 0, 255)
        intersected_image = cv.bitwise_or(dilated_image, inverted_binary_image)
        if np.array_equal(intersected_image, binary_aux):
            break
        binary_aux = intersected_image
        i += 1

    # new_image = np.array(binary_aux, dtype=np.uint8)
    # dilated_image = cv.erode(new_image, kernel)
    # cv.imwrite("..\\output\\binary_aux_image_1.png", dilated_image)

    # return union(binary_aux, binary_image, 0, 255)
    return cv.bitwise_and(binary_image, binary_aux)

start = time.time()
after_flood_fill = flood_fill(raw_image, 48, 111)
end = time.time()
cv.imwrite("..\\output\\after_flood_fill.png", after_flood_fill)

print("execution time in seconds: ", end - start)

