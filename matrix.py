##符号计算变换矩阵及其导数，结果直接应用在realign.py中
import sympy
import numpy as np
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
    coor = sym_v2d()
    M = np.array(coor.inv() @ M_r @ coor)
    return M

def sym_v2d():

    sym_descartes = sympy.Matrix([[affine_0, 0, 0, -0.5*affine_0*shape_0],
                                  [0, affine_1, 0, -0.5*affine_1*shape_1],
                                  [0, 0, affine_2, -0.5*affine_2*shape_2],
                                  [0, 0, 0, 1]])
    return sym_descartes



(affine_0, affine_1, affine_2) = sympy.symbols('affine_0:3')
(shape_0, shape_1, shape_2) = sympy.symbols('shape_0:3')
(q_0, q_1, q_2, q_3, q_4, q_5, q_6) = sympy.symbols('q_0:7')
sym_M = sym_rigid()


diff_1 = [1, 0, 0]  # 数值就是这个，节约计算时间
diff_2 = [0, 1, 0]
diff_3 = [0, 0, 1]
diff_4 = [sympy.diff(np.sum(sym_M[0]), q_4), sympy.diff(
    np.sum(sym_M[1]), q_4), sympy.diff(np.sum(sym_M[2]), q_4)]
# print(diff_4)
diff_5 = [sympy.diff(np.sum(sym_M[0]), q_5), sympy.diff(
    np.sum(sym_M[1]), q_5), sympy.diff(np.sum(sym_M[2]), q_5)]
diff_6 = [sympy.diff(np.sum(sym_M[0]), q_6), sympy.diff(
    np.sum(sym_M[1]), q_6), sympy.diff(np.sum(sym_M[2]), q_6)]
diff_0 = [1, 1, 1]  # 占个位置先
diff = np.array([diff_1, diff_2, diff_3, diff_4,
                diff_5, diff_6, diff_0])
