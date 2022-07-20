#!/usr/local/bin/python2.7
# python search.py -i dataset/train/ukbench00000.jpg
import pandas as pd
import argparse as ap
import cv2
# import imutils
import numpy as np
import os
import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
import numpy as np

from pylab import *
from PIL import Image
Q=np.array([[ 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,-6.52952347e+02],
            [ 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,-3.56892712e+02],
            [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.27717947e+02],
            [ 0.00000000e+00, 0.00000000e+00, 8.32664613e-03,-0.00000000e+00]])

orb = cv2.ORB_create(1000)

MIN_MATCH_COUNT = 6  # Number of minimum required feature points

good_match_rate = 0.1
good_match_rate1 = 0.02
bf = cv2.BFMatcher()
    # Load the classifier, class names, scaler, number of clusters and vocabulary
im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")

def ORBmatch(image_path):
    des_list = []
    k = image_path.split('/')[5]
    k = int(int(k.split('.')[0]) / 4)
    im = cv2.imread(image_path)
    kpts = orb.detect(im)
    kpts, des = orb.compute(im, kpts)



    des_list.append((image_path, des))

        # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]

        #
    test_features = np.zeros((1, numWords), "float32")
    words, distance = vq(descriptors, voc)
    for w in words:
        test_features[0][w] += 1

        # Perform Tf-Idf vectorization and L2 normalization
    test_features = test_features * idf
    test_features = preprocessing.normalize(test_features, norm='l2')

    score = np.dot(test_features, im_features.T)
    rank_ID = np.argsort(-score)

        # Visualize the results
    # figure()
    # gray()
    # subplot(5, 4, 1)
    # # imshow(im[:, :, ::-1])
    # axis('off')
    IN = []
    for i, ID in enumerate(rank_ID[0][0:5]):
        AA =  image_paths[ID].split('/')[5]
        IN.append(int(AA.split('.')[0]))
    IN.append(int(k))
    return IN



def draw(img1, kp1, img2, kp2, match):

    if len(match) >= MIN_MATCH_COUNT:

        src_pts = np.float32([kp1[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)

        dst_pts = np.float32([kp2[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,25.0)

        matchesMask = mask.ravel().tolist()

        # h, w = img1.shape
        #
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        #
        # dst = cv2.perspectiveTransform(pts, M)

        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)



    else:

        print("Not enough matches are found - %d/%d") % (len(match), MIN_MATCH_COUNT)

        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color

                        singlePointColor=None,

                        # matchesMask=matchesMask,  # draw only inliers

                        flags=2)
    count=0
    mat=[]
    for i in matchesMask:
        if i==1:
            mat.append(match[count])
        count=count+1
    # print(mat)
    # out = cv2.drawMatches(img1, kp1, img2, kp2, mat, outImg=None, **draw_params)
    # cv2.imshow("pipei",out)
    # cv2.waitKey(100)
    return mat

def ORB_Feature(img1,img2):
    kp1, des1 = orb.detectAndCompute(img1, None)

    kp2, des2 = orb.detectAndCompute(img2, None)
    # print(type(des1))


    # print(type(des1[0][0]),type(des2[0][0]))
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    good = matches[:int(len(matches) * good_match_rate)]
    good = draw(img1,kp1,img2,kp2,good)
    desT=[]
    kp1T=[]
    for x in good:
        # a = x.queryIdx
        # print(a)
        desT.append(des1[x.queryIdx].tolist())
        kp1T.append(kp1[x.queryIdx])
        # print(desT)
    # print(desT)
    desT = np.array(desT)
    return kp1,kp2,desT,good,kp1T

# def Mapdata():
#     path = "D:/Desktop/LV/qq/left/"
#     P = os.listdir(path)
#     P.sort(key=lambda x: int(x.split('.')[0]))
#     path_list = []
#     BBB=[]
#     kpTT=[]
#     desTT=[]
#     count = 0
#     for filename in P:
#         path_list.append(os.path.join(path, filename))
#     for left in path_list:
#         iml = cv2.imread(left)
#         imr = cv2.imread(left.replace('left', 'right'))
#         kp1, des1 = orb.detectAndCompute(iml, None)
#         kp2, des2 = orb.detectAndCompute(imr, None)
#         matches = bf.match(des1, des2)
#         matches = sorted(matches, key=lambda x: x.distance)
#         good = matches[:int(len(matches) * good_match_rate)]
#         good = draw(iml, kp1, imr, kp2, good)
#         BB=[]
#         desT = []
#         kpT = []
#         for x in good:
#             count = count + 1
#             a = x.queryIdx
#             kpT.append(kp1[a])
#         kpTT.append(kpT)
#     return kpTT
