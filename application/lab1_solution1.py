"""
1. Write an application that loads two images:

    a.       Scene image
    b.      Logo image

And superposes the logo image over the scene and allows to see through the zones in the logo that do not contain
    details/information. Hint: use the opencv_logo.png as logo
"""

import cv2 as cv
import sys

raw_logo_image = cv.imread(cv.samples.findFile("..\\resources\\open-cv-logo.png"))
raw_main_image = cv.imread(cv.samples.findFile("..\\resources\\Nevastuica-zboara-pe-ciocanitoare.jpg"))

# cv.imshow("Main image", raw_main_image)
# k = cv.waitKey(0)

# resize the logo image:
logo_image_resized = cv.resize(raw_logo_image, dsize=(60, 60), interpolation = cv.INTER_AREA)

# get the grayscale version of the logo image:
grayscale_logo_image = cv.cvtColor(logo_image_resized, cv.COLOR_BGR2GRAY)

# apply the thresholding:
ret, binary_logo_image = cv.threshold(grayscale_logo_image, 0, 255,  cv.THRESH_BINARY)

roi = raw_main_image[0:binary_logo_image.shape[0], raw_main_image.shape[1]-
                                                   binary_logo_image.shape[0]:raw_main_image.shape[1]]
cv.add(logo_image_resized, roi, roi, binary_logo_image)

cv.imshow("Grayscale Image", raw_main_image)
k = cv.waitKey(0)
