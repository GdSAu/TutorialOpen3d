import numpy as np
import cv2
folder = ('astra/cuadro/')
img = cv2.imread(folder + 'captura_10.png')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h)) # return a refined camera matrix and a Region of Interest
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imshow("Undistorted image", dst) # display
cv2.waitKey(0)
