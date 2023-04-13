import nibabel as nib
import matplotlib.pyplot as plt

# 导入NII
path = "C:\\Users\\陈穆方\\Desktop\\python程序设计\\项目文件\\data\\data\\sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii\\sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii"
img = nib.load(path)
img_data = img.get_fdata()
shape = img_data.shape#(96, 30, 64, 200)
slize0=img_data[42,:,:,0]
slize1=img_data[:,15,:,0]
slize2=img_data[:,:,32,0]
fig,a=plt.subplots(1,3)
plt.subplot(131)
plt.imshow(slize0,cmap="gray")
plt.title('0')
plt.subplot(132)
plt.imshow(slize1,cmap="gray")
plt.title('1')
plt.subplot(133)
plt.imshow(slize2,cmap="gray")
plt.title('2')
plt.show()
