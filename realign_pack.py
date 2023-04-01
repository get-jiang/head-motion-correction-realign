import nibabel as nib
import numpy as np
from math import cos, sin
from scipy.interpolate import *


class Realign:
    def __init__(self, path):
        self.path = path
        self.img = nib.load(path)
        self.img_data = self.img.get_fdata()
        self.shape = self.img_data.shape


# v2d也需要符号式的改写！！！！！！！！！！！
#但是这里没有
    def v2d(self,shape, affine):
        # 体素坐标到笛卡尔坐标的转换,并将坐标原点移到图像中心
        # voxel to descartes coordinate system
        # Move the coordinate origin to the image center
        descartes = np.zeros((4, 4))
        descartes[:, 0:3] = affine[:, 0:3]
        shape1 = np.asarray(shape)
        shape1[3] = 0
        descartes[:, 3] = -0.5 * shape1 + affine[:, 3]
        return descartes
    # numpy版


    def rigid(self, q):
        # 刚体变换/Rigid body transformation

        # 平移矩阵/translation matrix
        T = np.array([[1, 0, 0, q[1]],
                    [0, 1, 0, q[2]],
                    [0, 0, 1, q[3]],
                    [0, 0, 0, 1]])
        # 旋转矩阵/Rotation matrix
        R_x = np.array([[1, 0, 0, 0],
                        [0, cos(q[4]), sin(q[4]), 0],
                        [0, -sin(q[4]), cos(q[4]), 0],
                        [0, 0, 0, 1]])
        R_y = np.array([[cos(q[5]), 0, sin(q[5]), 0],
                        [0, 1, 0, 0],
                        [-sin(q[5]), 0, cos(q[5]), 0],
                        [0, 0, 0, 1]])
        R_z = np.array([[cos(q[6]), sin(q[6]), 0, 0],
                        [-sin(q[6]), cos(q[6]), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        R = R_x @ R_y @ R_z
        M_r = T @ R
        coor = Realign.v2d(self,self.shape, self.img.affine)
        M = np.linalg.inv(coor) @ M_r @ coor
        return M

        


    def LS(self, resource, reference,q):
        # 创建网格为插值铺垫
        x = np.arange(self.shape[0])
        y = np.arange(self.shape[1])
        z = np.arange(self.shape[2])

        # 残差函数--最小二乘法  非常慢，需要优化
        # B样条插值/B-spline interpolation
        # 因为找不到三维bspline及其导数的简单库，先用regular grid interpolator代替bspline,用切线斜率代替导数
        interp = RegularGridInterpolator((x, y, z), resource, method="quintic")
        # 对应位置的转换
        step = 20  # 因为跑的慢所以不要每个点都跑，先选一些试试
        bi = np.zeros(((self.shape[0]//step)*(self.shape[1]//step)*(self.shape[2]//step)))
        b_diff = np.zeros(
            ((self.shape[0]//step)*(self.shape[1]//step)*(self.shape[2]//step), 7))  # 偏导矩阵
        index = 0
        for i in range(0, self.shape[0]//step):
            for j in range(0, self.shape[1]//step):
                for k in range(0, self.shape[2]//step):
                    n_i = i*step  # 在原图的位置
                    n_j = j*step
                    n_k = k*step
                    M = Realign.rigid(self,q)
                    Mi = M@[n_i, n_j, n_k, 1]
                    
                    if 0 <= Mi[0] < self.shape[0] and 0 <= Mi[1] < self.shape[1] and 0 <= Mi[2] < self.shape[2]:
                        Mi_near = [i+1 for i in Mi]
                        bi[index] = interp(Mi[:3])-reference[n_i][n_j][n_k]
                        # 切线近似偏导数，分母为1省略
                        diff_x = float(
                            interp(np.append(Mi_near[0], Mi[1:3]))-interp(Mi[:3]))
                        diff_y = float(
                            interp(np.append(np.append(Mi[0], Mi_near[1]), Mi[2]))-interp(Mi[:3]))
                        diff_z = float(
                            interp(np.append(Mi[0:2], Mi_near[2]))-interp(Mi[:3]))

                    else:
                        bi[index] = resource[n_i][n_j][n_k] - \
                            reference[n_i][n_j][n_k]  # 超出范围就不转了
                        Mi_near = [n_i+1,n_j+1,n_k+1]
                        diff_x = float(
                            interp(np.append(Mi_near[0],[ n_j,n_k]))-resource[n_i][n_j][n_k])
                        diff_y = float(
                            interp(np.append(np.append([n_i], Mi_near[1]),[n_k]))-resource[n_i][n_j][n_k])
                        diff_z = float(
                            interp(np.append([n_i,n_j], Mi_near[2]))-resource[n_i][n_j][n_k])
                    ##很丑的代码
                    b_diff[index][0] = diff_x
                    b_diff[index][1] = diff_y
                    b_diff[index][2] = diff_z
                    #b_diff[index][3] = diff_y*(-10.33*sin(q[4])*sin(q[6]) + 2.33*sin(q[4])*cos(q[6]) + 2.33*sin(q[5])*sin(q[6])*cos(q[4]) + 10.33*sin(q[5])*cos(q[4])*cos(q[6]) - 6.77*cos(q[4])*cos(
                            #q[5])) + diff_z*(-6.97*sin(q[4])*sin(q[5])*sin(q[6]) - 30.90*sin(q[4])*sin(q[5])*cos(q[6]) + 20.26*sin(q[4])*cos(q[5]) - 30.90*sin(q[6])*cos(q[4]) + 6.97*cos(q[4])*cos(q[6]))
                    #b_diff[index][4] = diff_x*(7.0*sin(q[5])*sin(q[6]) + 31.0*sin(q[5])*cos(q[6]) - 20.33*cos(q[5]))+diff_y*(6.77*sin(q[4])*sin(q[5]) + 2.33*sin(q[4])*sin(q[6])*cos(
                            #q[5]) + 10.33*sin(q[4])*cos(q[5])*cos(q[6])) + diff_z*(20.26*sin(q[5])*cos(q[4]) + 6.97*sin(q[6])*cos(q[4])*cos(q[5]) + 30.906*cos(q[4])*cos(q[5])*cos(q[6]))
                    #b_diff[index][5] = diff_x*(31.0*sin(q[6])*cos(q[5]) - 7.0*cos(q[5])*cos(q[6]))+diff_y*(-10.33*sin(q[4])*sin(q[5])*sin(q[6]) + 2.33*sin(q[4])*sin(q[5])*cos(q[6]) + 2.33*sin(q[6])*cos(
                            #q[4]) + 10.33*cos(q[4])*cos(q[6]))+diff_z*(-6.97*sin(q[4])*sin(q[6]) - 30.90*sin(q[4])*cos(q[6]) - 30.90*sin(q[5])*sin(q[6])*cos(q[4]) + 6.97*sin(q[5])*cos(q[4])*cos(q[6]))
                    b_diff[index][3] = diff_y*(-0.5*self.img.affine[0][0]*self.shape[0]*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.img.affine[1][1] + self.img.affine[0][0]*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.img.affine[1][1] - 0.5*self.shape[1]*(-sin(q[4])*cos(q[6]) - sin(q[5])*sin(q[6])*cos(q[4])) - sin(q[4])*cos(q[6]) - sin(q[5])*sin(q[6])*cos(q[4]) - 0.5*self.img.affine[2][2]*self.shape[2]*cos(q[4])*cos(q[5])/self.img.affine[1][1] + self.img.affine[2][2]*cos(q[4])*cos(q[5])/self.img.affine[1][1]) + \
                            diff_z*(-0.5*self.img.affine[0][0]*self.shape[0]*(sin(q[4])*sin(q[5])*cos(q[6]) + sin(q[6])*cos(q[4]))/self.img.affine[2][2] + self.img.affine[0][0]*(sin(q[4])*sin(q[5])*cos(q[6]) + sin(q[6])*cos(q[4]))/self.img.affine[2][2] - 0.5*self.img.affine[1][1]*self.shape[1]*(
                            sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.img.affine[2][2] + self.img.affine[1][1]*(sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.img.affine[2][2] + 0.5*self.shape[2]*sin(q[4])*cos(q[5]) - sin(q[4])*cos(q[5]))

                    b_diff[index][4] = diff_x*(0.5*self.shape[0]*sin(q[5])*cos(q[6]) - sin(q[5])*cos(q[6]) + 0.5*self.img.affine[1][1]*self.shape[1]*sin(q[5])*sin(q[6])/self.img.affine[0][0] - self.img.affine[1][1]*sin(q[5])*sin(q[6])/self.img.affine[0][0] - 0.5*self.img.affine[2][2]*self.shape[2]*cos(q[5])/self.img.affine[0][0] + self.img.affine[2][2]*cos(q[5])/self.img.affine[0][0]) +\
                            diff_y*(0.5*self.img.affine[0][0]*self.shape[0]*sin(q[4])*cos(q[5])*cos(q[6])/self.img.affine[1][1] - self.img.affine[0][0]*sin(q[4])*cos(q[5])*cos(q[6])/self.img.affine[1][1] + 0.5*self.shape[1]*sin(q[4])*sin(q[6])*cos(q[5]) - sin(q[4])*sin(q[6])*cos(q[5]) + 0.5*self.img.affine[2][2]*self.shape[2]*sin(q[4])*sin(q[5])/self.img.affine[1][1] - self.img.affine[2][2]*sin(q[4])*sin(q[5])/self.img.affine[1][1]) +\
                            diff_z*(0.5*self.img.affine[0][0]*self.shape[0]*cos(q[4])*cos(q[5])*cos(q[6])/self.img.affine[2][2] - self.img.affine[0][0]*cos(q[4])*cos(q[5])*cos(q[6])/self.img.affine[2][2] + 0.5*self.img.affine[1][1]*self.shape[1]*sin(
                            q[6])*cos(q[4])*cos(q[5])/self.img.affine[2][2] - self.img.affine[1][1]*sin(q[6])*cos(q[4])*cos(q[5])/self.img.affine[2][2] + 0.5*self.shape[2]*sin(q[5])*cos(q[4]) - sin(q[5])*cos(q[4]))

                    b_diff[index][5] = diff_x*(0.5*self.shape[0]*sin(q[6])*cos(q[5]) - sin(q[6])*cos(q[5]) - 0.5*self.img.affine[1][1]*self.shape[1]*cos(q[5])*cos(q[6])/self.img.affine[0][0] + self.img.affine[1][1]*cos(q[5])*cos(q[6])/self.img.affine[0][0]) +\
                            diff_y*(-0.5*self.img.affine[0][0]*self.shape[0]*(sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.img.affine[1][1] + self.img.affine[0][0]*(sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.img.affine[1][1] - 0.5*self.shape[1]*(-sin(q[4])*sin(q[5])*cos(q[6]) - sin(q[6])*cos(q[4])) - sin(q[4])*sin(q[5])*cos(q[6]) - sin(q[6])*cos(q[4])) +\
                            diff_z*(-0.5*self.img.affine[0][0]*self.shape[0]*(sin(q[4])*cos(q[6]) + sin(q[5])*sin(q[6])*cos(q[4]))/self.img.affine[2][2] + self.img.affine[0][0]*(sin(q[4])*cos(q[6]) + sin(q[5])*sin(q[6])*cos(q[4]))/self.img.affine[2]
                            [2] - 0.5*self.img.affine[1][1]*self.shape[1]*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.img.affine[2][2] + self.img.affine[1][1]*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.img.affine[2][2])


                    b_diff[index][6] = -reference[n_i][n_j][n_k]
                    index += 1
        return bi,b_diff


    def perform(self):
        q = np.zeros(7, np.float64)
        
        # 高斯牛顿迭代
        # 算残差函数
        b_total = []
        for picture in range(2, self.shape[3]):
            for i in range(4):
                (b,diff_b) = Realign.LS(self, self.img_data[:, :, :, 1], self.img_data[:, :, :, picture],q)
                q -=np.linalg.inv(diff_b.T@diff_b)@diff_b.T@b
                #print(q)
                #print(sum(b))
                b_total.append(sum(b))
            return str(b_total)