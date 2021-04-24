import cv2
import numpy as np

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

# Change to gray scale to convert it from multi-channel (RGB) to single channel image which is easier to process
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
#cv2.imshow('result', gray)

# Gaissian filter
# Kernel of normally distributed values. It takes the weighted average of each pixel with the neighbouring pixels, thus smoothing the lane_image
# 5 by 5 is a typical kernal size
# GaussianBlur is used to reduce noise in the gray scale image
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow('result', blur)

# Apply "Canny" method to identify edges in our image. Done through identifying change in COLOR
# Change in brightness is called Gradient
# Canny function calculates the derivative of the image function f(x,y) calculating the gradient which tells us the change in brightness
# Canny outlines the strongest gradients of our image
canny = cv2.Canny(blur, 50, 150) #50 = low threshold, 150=high threshold
cv2.imshow('result', canny)

cv2.waitKey(0)
