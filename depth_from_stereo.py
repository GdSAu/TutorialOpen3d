import cv2
import numpy as np

# Disparity: the depth of a point in a scene is inversely proportional to the difference
# in distance of corresponding image points and their camera centers

imgL = cv2.imread("Tsukuba_L.png", cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread("Tsukuba_R.png", cv2.IMREAD_GRAYSCALE)

scale = 1.0             # scale factor for windows

wL = int(imgL.shape[1] * scale)     # left image width
hL = int(imgL.shape[0] * scale)     # left image height
wR = int(imgR.shape[1] * scale)
hR = int(imgR.shape[0] * scale)

# windows coordinates
x0, y0 = 0, 0
x1, y1 = x0 + wL, y0
x2, y2 = x1 + wR, y0


cx = wL / 2
cy = hL / 2

minDisparity = 0
numDisparities = 16
blockSize = 21

# Number of disparities: How many pixels to slide the window over.
# The larger it is, the larger the range of visible depths, but more computation is required.
# min_disparity: the offset from the x-position of the left pixel at which to begin searching.

# StereoBM: Class for computing stereo correspondence using the block matching algorithm
stereo = cv2.StereoBM.create(numDisparities, blockSize)
# compute: Computes disparity map for the specified stereo pair. 
disparity = stereo.compute(imgL, imgR)
# normalize: returns the normalized image
disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# window size for depth map
wD = int(disparity_vis.shape[1] * scale)
hD = int(disparity_vis.shape[0] * scale)

cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
cv2.imshow("Left", imgL)
cv2.imshow("Right", imgR)
cv2.imshow("Disparity", disparity_vis)
cv2.resizeWindow("Left", wL, hL)
cv2.resizeWindow("Right", wR, hR)
cv2.resizeWindow("Disparity", wD, hD)
cv2.moveWindow("Left", x0, y0)
cv2.moveWindow("Right", x1, y1)
cv2.moveWindow("Disparity", x2, y2)

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
