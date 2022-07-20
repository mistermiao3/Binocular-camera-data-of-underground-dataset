import numpy as np


####################仅仅是一个示例###################################


# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[527., 0, 655],
                                         [0., 527, 350],
                                         [0., 0., 1.]])
        # 右相机内参
        self.cam_matrix_right = np.array([[526., 0, 641],
                                          [0., 525., 362],
                                          [0., 0., 1.]])
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.0414,0.0109, 0.0006,0.0005]])
        self.distortion_r = np.array([[-0.0414,0.0109, 0.0006,0.0005]])
        # 旋转矩阵
        self.R = np.array([[1.0, 0, 0],
                           [0, 1.0, 0],
                           [0, 0, 1.0]])
        # 平移矩阵
        self.T = np.array([[-120.0960], [-0.0642], [0.2924]])
        # 焦距
        self.focal_length = 527.7179467297101  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]
        # 基线距离
        self.baseline = 120.0960 # 单位：mm， 为平移向量的第一个参数（取绝对值）
