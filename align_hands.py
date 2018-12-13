import cv2
import numpy as np
import pdb

# the goal of this script is to align two binary masks of hands, to maintain the location of a point on one of the w.r.t. the new location

mask1 = cv2.imread("mask1.png")
mask2 = cv2.imread("mask2.png")
#this needs to be an actual 2D array and not simply a view into it
idx1 = cv2.findContours(mask1[...,0].copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1][0]
idx2 = cv2.findContours(mask2[...,0].copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1][0]
contour1 = np.zeros(mask1.shape[:2])
contour2 = np.zeros(mask1.shape[:2])

cv2.drawContours(contour1, idx1, -1, 255, 3)
cv2.drawContours(contour2, idx2, -1, 255, 3)
cv2.imshow("contour1", contour1)
cv2.imshow("contour2", contour2)

cv2.waitKey(10)

# Specify the number of iterations.
number_of_iterations = 5000;
 
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10;
#homography
warp_matrix = np.eye(3, 3, dtype=np.float32)

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
warp_mode = cv2.MOTION_HOMOGRAPHY
# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC (contour1.astype(np.uint8),contour2.astype(np.uint8),warp_matrix, warp_mode, criteria)


pdb.set_trace()
if warp_mode == cv2.MOTION_HOMOGRAPHY:
    contour2_aligned = cv2.warpPerspective (contour2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    cv2.imshow("warped", contour2_aligned)
    cv2.waitKey(1)
