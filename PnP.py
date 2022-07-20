import numpy as np
import os
import cv2

from matplotlib import pyplot as plt
import stereoconfig


def Localization(a):
    A = np.array(a, dtype=np.double)
    img_point = a[:, [3, 4]]
    mdl_point = a[:, [0, 1, 2]]
    # print(mdl_point)
    cmr = np.array([[1054.9900, 0, 993.5800],
                    [0., 1054.5900, 524.3590],
                    [0., 0., 1.]])                                      #相机内参
    distCoeffs = np.array([[-0.0414, 0.0109, 0.0006, 0.0005, -0.0051]]) #畸变参数
    # print(cmr)
    S, R, Ta, = cv2.solvePnP(mdl_point, img_point, cmr, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    # print(S, R, T)
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(mdl_point, img_point, cmr, distCoeffs,
                                                     flags=cv2.SOLVEPNP_ITERATIVE)
    rotM = cv2.Rodrigues(R)[0]
    rotM1 = cv2.Rodrigues(rvec)[0]

    Ta = np.dot(-np.matrix(rotM).T, np.matrix(Ta))
    tvec = np.dot(-np.matrix(rotM1).T, np.matrix(tvec))

