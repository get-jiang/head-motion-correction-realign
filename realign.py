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


def LS(resource, reference):
    # 残差函数--最小二乘法  非常慢，需要优化
    # B样条插值/B-spline interpolation
    # 因为找不到三维bspline及其导数的简单库，先用regular grid interpolator代替bspline,用切线斜率代替导数
    interp = RegularGridInterpolator((x, y, z), resource, method="quintic")
    # 对应位置的转换

    bi = np.zeros(((shape[0]//step)*(shape[1]//step)*(shape[2]//step)))
    b_diff = np.zeros(
        ((shape[0]//step)*(shape[1]//step)*(shape[2]//step), 7))  # 偏导矩阵
    index = 0
    for i in range(0, shape[0]//step):
        for j in range(0, shape[1]//step):
            for k in range(0, shape[2]//step):
                n_i = i*step  # 在原图的位置
                n_j = j*step
                n_k = k*step
                Mi = M@[n_i, n_j, n_k, 1]
                Mi_near = [i+1 for i in Mi]
                if 0 <= Mi[0] < shape[0] and 0 <= Mi[1] < shape[1] and 0 <= Mi[2] < shape[2]:
                    bi[index] = interp(Mi[:3])-reference[n_i][n_j][n_k]
                    # 切线近似偏导数，分母为1省略
                    diff_x = float(
                        interp(np.append(Mi_near[0], Mi[1:3]))-interp(Mi[:3]))
                    diff_y = float(
                        interp(np.append(np.append(Mi[0], Mi_near[1]), Mi[2]))-interp(Mi[:3]))
                    diff_z = float(
                        interp(np.append(Mi[0:2], Mi_near[2]))-interp(Mi[:3]))
                    for iindex in range(6):
                        print(diff[iindex+3][0])
                        
                        b_diff[index][iindex] = diff_x*(diff[iindex][0].subs('q_'+iindex, q[iindex])) + \
                            diff_y*(diff[iindex][1].subs('q_'+iindex, q[iindex])) + \
                            diff_z*(diff[iindex]
                                    [2].subs('q_'+iindex, q[iindex]))
                else:
                    bi[index] = resource[n_i][n_j][n_k] - \
                        reference[n_i][n_j][n_k]  # 超出范围就不转了
                    for iindex in range(6):
                        b_diff[index][iindex] = diff_x*diff[iindex][0] + \
                            diff_y*diff[iindex][1]+diff_z*diff[iindex][2]
                b_diff[index][6] = -reference[n_i][n_j][n_k]
                index += 1
    print(b_diff)
    return bi


# -----------------------------------------------
# 读取数据/load data
path = "C:\\Users\\32385\\Desktop\\data\\sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii.gz"
img = nib.load(path)
img_data = img.get_fdata()
shape = img_data.shape
# ------------------------------------------------
# 初始化刚体变换向量及变换矩阵(为方便理解，q[0]为灰度平衡参数q7，目前版本中暂不使用,默认为1)
# Rigid body transformation vector
# (for easy understanding, q [0] is the gray balance parameter q7，not used in this version)
q = np.zeros(7, np.float64)
M = rigid(q)

# 创建网格为插值铺垫
x = np.arange(shape[0])
y = np.arange(shape[1])
z = np.arange(shape[2])


# 对于q的符号偏导
(q_0, q_1, q_2, q_3, q_4, q_5, q_6) = sympy.symbols('q_0:7')
sym_M = sym_rigid()
diff_1 = [1, 0, 0]  # 数值就是这个，节约计算时间
diff_2 = [0, 1, 0]
diff_3 = [0, 0, 1]
diff_4 = [sympy.diff(np.sum(sym_M[0]), q_4), sympy.diff(
    np.sum(sym_M[1]), q_4), sympy.diff(np.sum(sym_M[2]), q_4)]
diff_5 = [sympy.diff(np.sum(sym_M[0]), q_5), sympy.diff(
    np.sum(sym_M[1]), q_5), sympy.diff(np.sum(sym_M[2]), q_5)]
diff_6 = [sympy.diff(np.sum(sym_M[0]), q_6), sympy.diff(
    np.sum(sym_M[1]), q_6), sympy.diff(np.sum(sym_M[2]), q_6)]
diff_0 = [1, 1, 1]  # 占个位置先
diff = np.array([diff_1, diff_2, diff_3, diff_4,
                diff_5, diff_6, diff_0])

# 高斯牛顿迭代
# 算残差函数
# 取一张图试试先
step = 10  # 因为跑的慢所以不要每个点都跑，先选一些试试
b = LS(img_data[:, :, :, 1], img_data[:, :, :, 2])
# print(b)
