import nibabel as nib
import numpy as np
import math
from scipy.interpolate import *
import sympy


def v2d(shape, affine):
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


def rigid(q):
    # 刚体变换/Rigid body transformation

    # 平移矩阵/translation matrix
    T = np.array([[1, 0, 0, q[1]],
                  [0, 1, 0, q[2]],
                  [0, 0, 1, q[3]],
                  [0, 0, 0, 1]])
    # 旋转矩阵/Rotation matrix
    R_x = np.array([[1, 0, 0, 0],
                    [0, math.cos(q[4]), math.sin(q[4]), 0],
                    [0, -math.sin(q[4]), math.cos(q[4]), 0],
                    [0, 0, 0, 1]])
    R_y = np.array([[math.cos(q[5]), 0, math.sin(q[5]), 0],
                    [0, 1, 0, 0],
                    [-math.sin(q[5]), 0, math.cos(q[5]), 0],
                    [0, 0, 0, 1]])
    R_z = np.array([[math.cos(q[6]), math.sin(q[6]), 0, 0],
                    [-math.sin(q[6]), math.cos(q[6]), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    R = R_x @ R_y @ R_z
    M_r = T @ R
    coor = v2d(shape, img.affine)
    M = np.linalg.inv(coor) @ M_r @ coor
    return M


def sym_rigid():
    # 刚体变换/Rigid body transformation

    # 平移矩阵/translation matrix
    T = np.array([[1, 0, 0, q_1],
                  [0, 1, 0, q_2],
                  [0, 0, 1, q_3],
                  [0, 0, 0, 1]])
    # 旋转矩阵/Rotation matrix
    R_x = np.array([[1, 0, 0, 0],
                    [0, sympy.cos(q_4), sympy.sin(q_4), 0],
                    [0, -sympy.sin(q_4), sympy.cos(q_4), 0],
                    [0, 0, 0, 1]])
    R_y = np.array([[sympy.cos(q_5), 0, sympy.sin(q_5), 0],
                    [0, 1, 0, 0],
                    [-sympy.sin(q_5), 0, sympy.cos(q_5), 0],
                    [0, 0, 0, 1]])
    R_z = np.array([[sympy.cos(q_6), sympy.sin(q_6), 0, 0],
                    [-sympy.sin(q_6), sympy.cos(q_6), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    R = R_x @ R_y @ R_z
    M_r = T @ R
    coor = v2d(shape, img.affine)
    M = np.linalg.inv(coor) @ M_r @ coor
    return M
# 残差函数  非常慢，需要优化(甚至没有跑完一遍因为太慢了)
def b(resource, reference):
    # 插值
    interp = RegularGridInterpolator((x, y, z), resource, method="quintic")
    # 对应位置的转换
    bi = np.zeros((shape[0], shape[1], shape[2]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                Mi = M@[i, j, k, 1]
                if 0<=Mi[0]<shape[0] and 0<=Mi[1]<shape[1] and 0<=Mi[2]<shape[2]:
                    bi[i][j][k] = interp(Mi[:3])-reference[i][j][k]
                else:
                    bi[i][j][k] = resource[i][j][k]-reference[i][j][k]
                print(i,j,k)

    return print(bi)


# -----------------------------------------------
# 读取数据/load data
path = "C:\\Users\\32385\\Desktop\\data\\sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii.gz"
img = nib.load(path)
img_data = img.get_fdata()
shape = img_data.shape
# ------------------------------------------------
# 初始化刚体变换向量(为方便理解，q[0]为灰度平衡参数q7，目前版本中暂不使用,默认为1)
# Rigid body transformation vector
# (for easy understanding, q [0] is the gray balance parameter q7，not used in this version)
q = np.zeros(7, np.float64)


# B样条插值/B-spline interpolation
# 因为找不到三维bspline及其导数的简单库，先用regular grid interpolator代替bspline,用切线斜率代替导数
x = np.arange(shape[0])
y = np.arange(shape[1])
z = np.arange(shape[2])
# 取一个测试，inter是插值点的灰度值
# interp = RegularGridInterpolator(
#     (x, y, z), img_data[:, :, :, 1], method="quintic")
# test = [10.5, 10.5, 10.5]
# inter = interp(test)
# print(inter)

# 对于q的符号偏导
M = rigid(q)
q_1 = sympy.Symbol('q_1')
q_2 = sympy.Symbol('q_2')
q_3 = sympy.Symbol('q_3')
q_4 = sympy.Symbol('q_4')
q_5 = sympy.Symbol('q_5')
q_6 = sympy.Symbol('q_6')
q_0 = sympy.Symbol('q_0')
sym_M = sym_rigid()
diff_1 = 1  # 数值就是这个，节约计算时间直接写为1
diff_2 = 1
diff_3 = 1
diff_4 = sympy.diff(np.sum(
    sym_M[0]), q_4)+sympy.diff(np.sum(sym_M[1]), q_4)+sympy.diff(np.sum(sym_M[2]), q_4)
diff_5 = sympy.diff(np.sum(
    sym_M[0]), q_5)+sympy.diff(np.sum(sym_M[1]), q_5)+sympy.diff(np.sum(sym_M[2]), q_5)
diff_6 = sympy.diff(np.sum(
    sym_M[0]), q_6)+sympy.diff(np.sum(sym_M[1]), q_6)+sympy.diff(np.sum(sym_M[2]), q_6)
diff_0 = 1  # 占个位置先
diff = np.array([diff_1, diff_2, diff_3, diff_4, diff_5, diff_6, diff_0])
# print(diff)

# 高斯牛顿迭代
b(img_data[:, :, :, 1], img_data[:, :, :, 2])
