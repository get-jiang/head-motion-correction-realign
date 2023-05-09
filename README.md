# 头动校正
## 介绍
本项目是一个基于python的头动校正程序，输入nii格式的头动数据，输出校正后的头动数据及校正参数。
## 功能函数
1. 以set_order为代表的参数调节函数，可调整校正和重采样过程中使用的插值函数、迭代次数、采样点间距
2. estimate函数，用于估计校正参数，输出校正参数并画出头动曲线
3. resling函数，用于校正头动数据，生成新的图像，自动保存为nii格式
具体使用方式见测验代码
## 可改进方向
1. 提高校正准确度
2. 优化速度

