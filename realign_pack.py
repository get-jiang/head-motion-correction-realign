import numpy as np
from math import cos, sin
import itk


class Realign:
    def __init__(self, path):
        self.path = path
        self.img = itk.imread(path)  # 读取数据/load data
        # 转换为numpy数组/convert to numpy array
        self.img_data = itk.GetArrayFromImage(self.img)
        self.shape = self.img_data.shape
        spacing = self.img.GetSpacing()  # 体素间距/voxel spacing
        spacing_array = np.array(spacing)
        diagonal_matrix = np.diag(spacing_array)
        self.affine = diagonal_matrix
        
    
    def v2d(self):
        # 体素坐标到笛卡尔坐标的转换,并将坐标原点移到图像中心
        # voxel to descartes coordinate system
        # Move the coordinate origin to the image center
        descartes = np.zeros((4, 4))
        descartes[:, :3] = self.affine[:, :3]
        shape1 = np.asarray(self.shape)
        shape1[3] = 0
        descartes[:, 3] = -0.5 * shape1 + self.affine[:, 3]
        return descartes

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
        coor = Realign.v2d(self)
        M = np.linalg.inv(coor) @ M_r @ coor
        return M

    def LS(self, resource, reference, q):

        # 残差函数--最小二乘法
        # B样条插值/B-spline interpolation
        interpolator = itk.BSplineInterpolateImageFunction.New(self.img)
        interpolator.SetSplineOrder(3)  # 三阶B样条插值,（可变参数）
        # 对应位置的转换
        step = 20  # 选点间隔(可变参数)
        bi = np.zeros(
            ((self.shape[0]//step)*(self.shape[1]//step)*(self.shape[2]//step)))
        b_diff = np.zeros(
            ((self.shape[0]//step)*(self.shape[1]//step)*(self.shape[2]//step), 7))  # 偏导矩阵
        index = 0
        for i in range(0, self.shape[0]//step):
            for j in range(0, self.shape[1]//step):
                for k in range(0, self.shape[2]//step):
                    n_i = i*step  # 在原图的位置
                    n_j = j*step
                    n_k = k*step
                    M = Realign.rigid(self, q)
                    interpo_pos = M@[n_i, n_j, n_k, 1]

                    if 0 <= interpo_pos[0] < self.shape[0] and 0 <= interpo_pos[1] < self.shape[1] and 0 <= interpo_pos[2] < self.shape[2]:  # 判断是否在范围内
                        point = itk.Point[itk.D, 4]()
                        point[0] = interpo_pos[0]
                        point[1] = interpo_pos[1]
                        point[2] = interpo_pos[2]
                        point[3] = interpo_pos[3]
                        bi[index] = interpolator.Evaluate(point)-reference[n_i][n_j][n_k]  # 残差函数

                    else:
                        bi[index] = resource[n_i][n_j][n_k] - reference[n_i][n_j][n_k]  # 超出范围就不转了
                        point = itk.Point[itk.D, 4]()
                        point[0] = n_i
                        point[1] = n_j
                        point[2] = n_k
                        point[3] = 1
                        
                    derivative = interpolator.EvaluateDerivative(point)
                    diff_x = derivative[0]
                    diff_y = derivative[1]
                    diff_z = derivative[2]
                    b_diff[index][:3] = diff_x, diff_y, diff_z
                    b_diff[index][3] = diff_y*(-0.5*self.affine[0][0]*self.shape[0]*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.affine[1][1] + self.affine[0][0]*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.affine[1][1] - 0.5*self.shape[1]*(-sin(q[4])*cos(q[6]) - sin(q[5])*sin(q[6])*cos(q[4])) - sin(q[4])*cos(q[6]) - sin(q[5])*sin(q[6])*cos(q[4]) - 0.5*self.affine[2][2]*self.shape[2]*cos(q[4])*cos(q[5])/self.affine[1][1] + self.affine[2][2]*cos(q[4])*cos(q[5])/self.affine[1][1]) + \
                        diff_z*(-0.5*self.affine[0][0]*self.shape[0]*(sin(q[4])*sin(q[5])*cos(q[6]) + sin(q[6])*cos(q[4]))/self.affine[2][2] + self.affine[0][0]*(sin(q[4])*sin(q[5])*cos(q[6]) + sin(q[6])*cos(q[4]))/self.affine[2][2] - 0.5*self.affine[1][1]*self.shape[1]*(
                            sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.affine[2][2] + self.affine[1][1]*(sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.affine[2][2] + 0.5*self.shape[2]*sin(q[4])*cos(q[5]) - sin(q[4])*cos(q[5]))

                    b_diff[index][4] = diff_x*(0.5*self.shape[0]*sin(q[5])*cos(q[6]) - sin(q[5])*cos(q[6]) + 0.5*self.affine[1][1]*self.shape[1]*sin(q[5])*sin(q[6])/self.affine[0][0] - self.affine[1][1]*sin(q[5])*sin(q[6])/self.affine[0][0] - 0.5*self.affine[2][2]*self.shape[2]*cos(q[5])/self.affine[0][0] + self.affine[2][2]*cos(q[5])/self.affine[0][0]) +\
                        diff_y*(0.5*self.affine[0][0]*self.shape[0]*sin(q[4])*cos(q[5])*cos(q[6])/self.affine[1][1] - self.affine[0][0]*sin(q[4])*cos(q[5])*cos(q[6])/self.affine[1][1] + 0.5*self.shape[1]*sin(q[4])*sin(q[6])*cos(q[5]) - sin(q[4])*sin(q[6])*cos(q[5]) + 0.5*self.affine[2][2]*self.shape[2]*sin(q[4])*sin(q[5])/self.affine[1][1] - self.affine[2][2]*sin(q[4])*sin(q[5])/self.affine[1][1]) +\
                        diff_z*(0.5*self.affine[0][0]*self.shape[0]*cos(q[4])*cos(q[5])*cos(q[6])/self.affine[2][2] - self.affine[0][0]*cos(q[4])*cos(q[5])*cos(q[6])/self.affine[2][2] + 0.5*self.affine[1][1]*self.shape[1]*sin(
                            q[6])*cos(q[4])*cos(q[5])/self.affine[2][2] - self.affine[1][1]*sin(q[6])*cos(q[4])*cos(q[5])/self.affine[2][2] + 0.5*self.shape[2]*sin(q[5])*cos(q[4]) - sin(q[5])*cos(q[4]))

                    b_diff[index][5] = diff_x*(0.5*self.shape[0]*sin(q[6])*cos(q[5]) - sin(q[6])*cos(q[5]) - 0.5*self.affine[1][1]*self.shape[1]*cos(q[5])*cos(q[6])/self.affine[0][0] + self.affine[1][1]*cos(q[5])*cos(q[6])/self.affine[0][0]) +\
                        diff_y*(-0.5*self.affine[0][0]*self.shape[0]*(sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.affine[1][1] + self.affine[0][0]*(sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.affine[1][1] - 0.5*self.shape[1]*(-sin(q[4])*sin(q[5])*cos(q[6]) - sin(q[6])*cos(q[4])) - sin(q[4])*sin(q[5])*cos(q[6]) - sin(q[6])*cos(q[4])) +\
                        diff_z*(-0.5*self.affine[0][0]*self.shape[0]*(sin(q[4])*cos(q[6]) + sin(q[5])*sin(q[6])*cos(q[4]))/self.affine[2][2] + self.affine[0][0]*(sin(q[4])*cos(q[6]) + sin(q[5])*sin(q[6])*cos(q[4]))/self.affine[2]
                                [2] - 0.5*self.affine[1][1]*self.shape[1]*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.affine[2][2] + self.affine[1][1]*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.affine[2][2])

                    b_diff[index][6] = -reference[n_i][n_j][n_k]
                    index += 1
        return bi, b_diff

    def perform(self):
        #随机初始化参数
        q = np.random.rand(7)
        q[6]=1
        iteration = 4#迭代次数(可调)
        # 高斯牛顿迭代
        # 算残差函数
        whole_pic=[]
        for picture in range(2, self.shape[3]):
            for i in range(iteration):
                (b, diff_b) = Realign.LS(self, self.img_data[:, :, :, 1], self.img_data[:, :, :, picture], q)
                
                if np.linalg.det(diff_b.T@diff_b)==0:
                    q-=np.linalg.pinv(diff_b.T@diff_b)@diff_b.T@b#这里用伪逆是因为有时候矩阵不可逆
                else:
                    q -= np.linalg.inv(diff_b.T@diff_b)@diff_b.T@b
            whole_pic.append(q)
            print(f"第{picture}张图片的参数为：{q}")
        return str(whole_pic)#不知道为什么这里要加上str

#测验代码，用于测试上面的类，勿删
realign = Realign('sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii.gz')
print(realign.perform())
