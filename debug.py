import numpy as np
test = np.array([[[2,3,4],
                  [3,4,5]],
                  [[4,5,6],
                   [5,6,7]]])
pos =[1.5,1.5,1.5]
# d          degree of bspline(default 2)
# n(x,y,z)          number of control points
# k_x,k_y,x_z          number of knot intervals
# vector_kx, vector_ky, vector_kz  specific intervals boundries (uniform)
# t(u,v,w)   target position(单位化以后的位置)
#weight不能变，所以要把t笛卡尔坐标单位化 dizzy~~~~~~~~~~~~~~
def point_bspline(t, matrix, d=2):
    (x, y, z) = matrix.shape
    k_x = d+x
    k_y = d+y
    k_z = d+z
    vector_kx = np.arange(0, 1, 1/k_x)
    vector_ky = np.arange(0, 1, 1/k_y)
    vector_kz = np.arange(0, 1, 1/k_z)
    value = 0
    for i in range(x):
        for j in range(y):
            for k in range(z):
                value +=matrix[i][j][k]*weight(i,k_x,t[0],vector_kx)*weight(j,k_y,t[1],vector_ky)*weight(k,k_z,t[2],vector_kz)
    return value

def weight(i, k, t, vector_k):
    if k == 1:
        return 1 if (t >= vector_k[i] and t < vector_k[i+1]) else 0
    else:
        return (t-vector_k[i])/(vector_k[i+k-1]-vector_k[i])*weight(i, k-1, t, vector_k)+(vector_k[i+k]-t)/(vector_k[i+k]-vector_k[i+1])*weight(i+1, k-1, t, vector_k)
print(point_bspline(pos,test))