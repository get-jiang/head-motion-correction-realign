import nibabel as nib
import numpy as np
from math import cos, sin
from scipy.interpolate import *
import sympy

# v2d也需要符号式的改写！！！！！！！！！！！
#但是这里没有


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


def LS(resource, reference,q):
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
                M = rigid(q)
                Mi = M@[n_i, n_j, n_k, 1]
                
                if 0 <= Mi[0] < shape[0] and 0 <= Mi[1] < shape[1] and 0 <= Mi[2] < shape[2]:
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
                b_diff[index][3] = diff_y*(-10.33*sin(q[4])*sin(q[6]) + 2.33*sin(q[4])*cos(q[6]) + 2.33*sin(q[5])*sin(q[6])*cos(q[4]) + 10.33*sin(q[5])*cos(q[4])*cos(q[6]) - 6.77*cos(q[4])*cos(
                        q[5])) + diff_z*(-6.97*sin(q[4])*sin(q[5])*sin(q[6]) - 30.90*sin(q[4])*sin(q[5])*cos(q[6]) + 20.26*sin(q[4])*cos(q[5]) - 30.90*sin(q[6])*cos(q[4]) + 6.97*cos(q[4])*cos(q[6]))
                b_diff[index][4] = diff_x*(7.0*sin(q[5])*sin(q[6]) + 31.0*sin(q[5])*cos(q[6]) - 20.33*cos(q[5]))+diff_y*(6.77*sin(q[4])*sin(q[5]) + 2.33*sin(q[4])*sin(q[6])*cos(
                        q[5]) + 10.33*sin(q[4])*cos(q[5])*cos(q[6])) + diff_z*(20.26*sin(q[5])*cos(q[4]) + 6.97*sin(q[6])*cos(q[4])*cos(q[5]) + 30.906*cos(q[4])*cos(q[5])*cos(q[6]))
                b_diff[index][5] = diff_x*(31.0*sin(q[6])*cos(q[5]) - 7.0*cos(q[5])*cos(q[6]))+diff_y*(-10.33*sin(q[4])*sin(q[5])*sin(q[6]) + 2.33*sin(q[4])*sin(q[5])*cos(q[6]) + 2.33*sin(q[6])*cos(
                        q[4]) + 10.33*cos(q[4])*cos(q[6]))+diff_z*(-6.97*sin(q[4])*sin(q[6]) - 30.90*sin(q[4])*cos(q[6]) - 30.90*sin(q[5])*sin(q[6])*cos(q[4]) + 6.97*sin(q[5])*cos(q[4])*cos(q[6]))
                b_diff[index][6] = -reference[n_i][n_j][n_k]
                index += 1
    return bi,b_diff


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
step = 20  # 因为跑的慢所以不要每个点都跑，先选一些试试
for picture in range(1,shape[3]):
    for i in range(4):
        (b,diff_b) = LS(img_data[:, :, :, 0], img_data[:, :, :, picture],q)
        q -=np.linalg.inv(diff_b.T@diff_b)@diff_b.T@b
        print(q)
        print(sum(b))