import cv2
import glob
import numpy as np
import os
orb = cv2.ORB_create()
# img_path = glob.glob("D:/Desktop/LV/qq/*.png")
bf = cv2.BFMatcher()

#输入查询图片地址，返回最近三个节点索引
def Lidarmatch(Lidardata,des_Lidar):

    # _,_,_,_,Quary = Lidardata.split('/')
    # Quary,_ = Quary.split('.')

    img1 = cv2.imread(Lidardata)
    kp1 = [cv2.KeyPoint(33,33,31.0)]
    # print(kp1)
    kp1,des1 =orb.compute(img1,kp1)
    matches = bf.knnMatch(des1, des_Lidar, k=5)

    A = matches[0][0].trainIdx
    B = matches[0][1].trainIdx
    C = matches[0][2].trainIdx
    D = matches[0][3].trainIdx
    E = matches[0][4].trainIdx
    k=Lidardata.split('/')[5]
    k =int(int(k.split('.')[0])/4)
    print(k)
    LN = [A,B,C,D,E,k]
    return LN
# 068BA7F0

def Lidarmatch1(Lidardata,des_Lidar,N):
    img1 = cv2.imread(Lidardata)
    kp1 = [cv2.KeyPoint(33, 33, 31.0)]
    # print(kp1)
    kp1, des1 = orb.compute(img1, kp1)
    i=0
    des=[]
    while i<=20:
        if N+i>=45:
            break
        des.append(des_Lidar[N+i].tolist())

        i = i + 1
    des = np.array(des,np.uint8)
    matches = bf.knnMatch(des1, des, k=5)
    A = matches[0][0].trainIdx + N
    B = matches[0][1].trainIdx + N
    C = matches[0][2].trainIdx + N
    D = matches[0][3].trainIdx + N
    E = matches[0][4].trainIdx + N
    k = Lidardata.split('/')[5]
    k = int(int(k.split('.')[0]) / 4)
    LN = [A, B, C, D, E,k]
    return LN

