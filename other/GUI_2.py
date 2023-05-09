from PyQt5.QtWidgets import *
from PyQt5 import uic
import sys
import nibabel as nib
import numpy as np
from realign_pack import Realign




class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi("GUI_1.ui", self)
        # define widgets
        self.button = self.findChild(QPushButton, "pushButton")
        self.button_2 = self.findChild(QPushButton, "pushButton_2")
        self.label = self.findChild(QLabel, "label")
        # click function
        self.button.clicked.connect(self.clicker)
        self.button_2.clicked.connect(self.clicker_2)
        # show app
        self.show()
        
    def clicker(self):
        open, _ = QFileDialog.getOpenFileName(self, "Open a file", "", "All files (*)")    
        # 读取数据/load data
        self.path = open
        self.img = nib.load(self.path)
        self.img_data = self.img.get_fdata()
        self.shape = self.img_data.shape
    
    def clicker_2(self):
        realign = Realign(self.path)
        
        self.label.setText(str(realign.estimate()))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    UIWindow = UI()
    sys.exit(app.exec_())    