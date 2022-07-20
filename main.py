import ORBmatch
import Lidarmatch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
orb = cv2.ORB_create(5000)
good_match_rate = 0.15
MIN_MATCH_COUNT = 6
bf = cv2.BFMatcher()

def path(path):
    path = path
    P = os.listdir(path)
    P.sort(key=lambda x: int(x.split('.')[0]))
    path_list = []
    for filename in P:
        path_list.append(os.path.join(path, filename))
    return path_list


if __name__ == '__main__':
    print("地图数据初始化")
    nd = np.genfromtxt('D:/Desktop/LV/qq/NodeLidar/1.csv', delimiter=',', skip_header=False)
    des_Lidar = np.array(nd.tolist(),np.uint8)
    dd = 0
    BBB=[]
    kpTT=[]
    desTT=[]
    while dd<=45:
        nd = np.genfromtxt('D:/Desktop/LV/qq/NodeImage/cood/%d.csv' %(dd), delimiter=',', skip_header=False)
        BB = nd.tolist()
        BBB.append(BB)
        nd = np.genfromtxt('D:/Desktop/LV/qq/NodeImage/kp/%d.csv' % (dd), delimiter=',', skip_header=False)
        kpt = nd.tolist()
        kpTT.append(kpt)
        nd = np.genfromtxt('D:/Desktop/LV/qq/NodeImage/des/%d.csv' % (dd), delimiter=',', skip_header=False)
        desTT.append(np.array(nd.tolist(),np.uint8))
        dd = dd+1

    print("开始轨迹计算")
    path_list = path("D:/Desktop/LV/qq/quaryimage/")
    count = 0
    NR = []
    fig = plt.figure(figsize=(16, 9), dpi=160)
    TTA = []
    for item in path_list:
        lidardata = item.replace('quaryimage', 'quarylidar')
        if count<10:
            LN = Lidarmatch.Lidarmatch(lidardata,des_Lidar)
            IN = ORBmatch.ORBmatch(item)

            res = list(set(LN) & set(IN))

            Node=res[0]
            NR.append(Node)
        else:
            LN = Lidarmatch.Lidarmatch1(lidardata, des_Lidar,NR[count%10])
            IN = ORBmatch.ORBmatch(item)
            res = list(set(LN) & set(IN))

            Node = res[-0]
            NR[count%10] = Node

        count = count + 1
        imageNode = "D:/Desktop/LV/qq/left/%d.png" %(Node)
        imq = cv2.imread(item)
        imn = cv2.imread(imageNode)
        kpq, desq = orb.detectAndCompute(imq, None)
        matches = bf.match(np.array(desTT[Node], np.uint8),desq)
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:int(len(matches) * good_match_rate)]
        if len(good) >= MIN_MATCH_COUNT:
            src_pts = np.float32([kpTT[Node][m.queryIdx] for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpq[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 25.0)
            matchesMask = mask.ravel().tolist()
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           # matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        c = 0
        mat = []
        for i in matchesMask:
            if i == 1:
                mat.append(good[c])
            c = c + 1
        kpn=[]
        for i in kpTT[Node]:
            kpn.append(cv2.KeyPoint(i[0], i[1], 31.0))
        out = cv2.drawMatches(imn, kpn, imq, kpq, mat, outImg=None, **draw_params)

        A=[]
        for x in mat:
            b = x.queryIdx
            a = x.trainIdx
            B = [BBB[Node][b][0],BBB[Node][b][1],BBB[Node][b][2],kpq[a].pt[0],kpq[a].pt[1]]
            A.append(B)
        A = np.array(A, dtype=np.double)
        img_point = A[:, [3, 4]]
        mdl_point = A = A[:, [0, 1, 2]]
        cmr = np.array([[527., 0, 655],
                        [0., 527, 350],
                        [0., 0., 1.]])
        distCoeffs = np.array([[-0.0414, 0.0109, 0.0006, 0.0005, -0.0051]])
        S, R, Ta, = cv2.solvePnP(mdl_point, img_point, cmr, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        rotM = cv2.Rodrigues(R)[0]
        Ta = np.dot(-np.matrix(rotM).T, np.matrix(Ta))
        TTA.append(Ta)
        sub1 = fig.add_subplot(111)
        sub1.imshow(out)
        sub1.set_xlabel('Closest Node from Map<<<----------------------------->>>Quary Image')
        mngr = plt.get_current_fig_manager()  # 获取当前figure manager
        mngr.window.wm_geometry("+0+0")
        plt.show(block=False)
        plt.pause(0.1)  # 显示秒数
        plt.clf()
        with open("D:/Desktop/LV/result.csv", 'w', newline='') as t:  # numline是来控制空的行数的
            writer = csv.writer(t)  # 这一步是创建一个csv的写入器（个人理解）
            # writer.writerow(b)  # 写入标签
            writer.writerows(TTA)  # 写入样本数据
