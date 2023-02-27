import nibabel as nib
import numpy as np
import math

# 体素坐标到笛卡尔坐标的转换,并将坐标原点移到图像中心
# voxel to descartes coordinate system
# Move the coordinate origin to the image center
def v2d(shape,affine):
    descartes = np.zeros((4,4))
    descartes[:,0:3] = affine[:,0:3]
    shape1 = np.asarray(shape)
    shape1[3] = 0
    descartes[:,3] = -0.5 * shape1 + affine[:,3]
    return descartes

#-----------------------------------------------
# 读取数据/load data
path = "C:\\Users\\32385\\Desktop\\data\\sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii.gz"
img = nib.load(path)
img_data = img.get_fdata()


#------------------------------------------------
#获取坐标转换矩阵/Get coordinate conversion matrix
coor = v2d(img_data.shape,img.affine)


#------------------------------------------------
## 刚体变换/Rigid body transformation

# 刚体变换向量(为方便理解，q[0]为灰度平衡参数q7)
# Rigid body transformation vector 
# (for easy understanding, q [0] is the gray balance parameter q7)
q = np.ones(7, np.float64)
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
M = np.linalg.inv(coor) @ M_r @ coor
