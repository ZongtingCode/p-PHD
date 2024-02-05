import os
import sys
from datetime import time

from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget, QProgressBar, QPushButton, QDialog, QMessageBox, QFileDialog, QApplication, \
    QMainWindow, QLabel, QVBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtCore import *
import csv
import h2o
import matplotlib.pyplot as plt
h2o.init()
import shap
shap.initjs()


class Stats:
    def __init__(self):
        # 从文件中加载UI定义
        self.ui = uic.loadUi("DDS-1920-1080-100.ui")

        # 信号槽设置
        self.ui.Click_For_Result.clicked.connect(self.ClickForResult)
        # 初始化设置
        self.ui.LiverDisease.setReadOnly(True)
        self.ui.Accuracy.setReadOnly(True)
        pix = QPixmap('img/test.png')
        self.ui.img.setPixmap(pix)
        self.ui.img.setScaledContents(True)

        self.ui.Risk.setEnabled(False)

    def ClickForResult(self):
        Patient_data = []
        # 输入校验，每条信息不能为空
        if self.ui.Age.displayText() == "":
            msgBox = QMessageBox()
            msgBox.setWindowTitle('Tips')
            msgBox.setText("Please enter the patient age!")
            msgBox.exec()
        elif self.ui.BUN.displayText() == "":
            msgBox = QMessageBox()
            msgBox.setWindowTitle('Tips')
            msgBox.setText("Please enter BUN!")
            msgBox.exec()
        elif self.ui.Albumin.displayText() == "":
            msgBox = QMessageBox()
            msgBox.setWindowTitle('Tips')
            msgBox.setText("Please enter the Albumin!")
            msgBox.exec()
        elif self.ui.HCT.displayText() == "":
            msgBox = QMessageBox()
            msgBox.setWindowTitle('Tips')
            msgBox.setText("Please enter the HCT!")
            msgBox.exec()
        elif self.ui.OperatingTime.displayText() == "":
            msgBox = QMessageBox()
            msgBox.setWindowTitle('Tips')
            msgBox.setText("Please enter the operating time!")
            msgBox.exec()
        elif self.ui.VascularResection.displayText() == "":
            msgBox = QMessageBox()
            msgBox.setWindowTitle('Tips')
            msgBox.setText("Please enter the vascular resection!")
            msgBox.exec()
        elif self.ui.Height.displayText() == "":
            msgBox = QMessageBox()
            msgBox.setWindowTitle('Tips')
            msgBox.setText("Please enter the Height!")
            msgBox.exec()
        elif self.ui.Cr.displayText() == "":
            msgBox = QMessageBox()
            msgBox.setWindowTitle('Tips')
            msgBox.setText("Please enter the Cr!")
            msgBox.exec()
        elif self.ui.INR.displayText() == "":
            msgBox = QMessageBox()
            msgBox.setWindowTitle('Tips')
            msgBox.setText("Please enter the INR!")
            msgBox.exec()
        elif self.ui.ASAclass.displayText() == "":
            msgBox = QMessageBox()
            msgBox.setWindowTitle('Tips')
            msgBox.setText("Please enter the total albumin and globulin ratio!")
            msgBox.exec()
        else:
            # 定义表头
            Patient_header = ['age', 'prbun', 'pralbum', 'prhct', 'optime', 'pan_vascular_resection', 'height',
                              'prcreat', 'prinr', 'ASAclass', 'othbleed']
            # 数据
            Patient_data.append(self.ui.Age.displayText())
            Patient_data.append(self.ui.BUN.displayText())
            Patient_data.append(self.ui.Albumin.displayText())
            Patient_data.append(self.ui.HCT.displayText())
            Patient_data.append(self.ui.OperatingTime.displayText())
            Patient_data.append(self.ui.VascularResection.displayText())
            Patient_data.append(self.ui.Height.displayText())
            Patient_data.append(self.ui.Cr.displayText())
            Patient_data.append(self.ui.INR.displayText())
            Patient_data.append(self.ui.ASAclass.displayText())
            print(Patient_data)

            with open('data/Patient.csv', 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(Patient_header)
                writer.writerow(Patient_data)
            # 获取模型数据
            Patient = h2o.import_file("data/Patient.csv")
            path = 'model/GBM_grid_1_AutoML_19_20231005_104801_model_2.zip'
            imported_model = h2o.import_mojo(path)
            predict = imported_model.predict(Patient)
            # 绘图
            contributions = imported_model.predict_contributions(Patient)
            contributions_df = contributions.as_data_frame()
            selected_row_index = 0  # 行索引从0开始
            # 获取对应索引的行数据
            sampled_contributions_df = contributions_df.iloc[[selected_row_index]]
            # Extract SHAP values for the sampled data
            shap_values_sampled = sampled_contributions_df.iloc[:, 0:10].values

            # Expected values is the last returned column
            expected_value = sampled_contributions_df.iloc[:, 10].mean()

            # Visualize the sampled data predictions using SHAP force plot
            X = ['Age', 'BUN', 'Albumin', 'HCT', 'Operative time', 'Vascular resection', 'Height', 'Cr', 'INR',
                 'ASA class']
            plt.figure(figsize=(15, 10))
            shap.force_plot(expected_value, shap_values_sampled, X, show=False, matplotlib=True)

            # 获取当前图形并保存
            plt.savefig('img/shap_force_plot_1.png', dpi=300, bbox_inches='tight')

            temp = predict.as_data_frame()
            auc = temp.values[0][0]
            LiverDisease = ""
            if temp.values[0][0]==1:
                LiverDisease = "YES"
                auc = temp.values[0][2]
                Patient_data.append(1)
            else:
                LiverDisease = "NO"
                auc = temp.values[0][2]
                Patient_data.append(0)
            with open('data/Patient.csv', 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(Patient_header)
                writer.writerow(Patient_data)

            if auc<=0.2:
                self.ui.Risk.setText("Low risk")
                self.ui.Risk.setStyleSheet("background-color: green; color: black")
            elif auc>0.2 and auc<0.5:
                self.ui.Risk.setText("Moderate risk")
                self.ui.Risk.setStyleSheet("background-color: yellow; color: black")
            elif auc>0.5 and auc<=1.0:
                self.ui.Risk.setText("High risk")
                self.ui.Risk.setStyleSheet("background-color: #780000; color: black")
            else:
                self.ui.Risk.setText("Risk")
            self.ui.LiverDisease.setText(LiverDisease)
            self.ui.Accuracy.setText(str(round(auc * 100, 2)) + "%")

            max_retries = 5
            current_retry = 0
            while not os.path.exists('img/shap_force_plot_1.png') and current_retry<max_retries:
                current_retry += 1
                time.sleep(1)
            if os.path.exists('img/shap_force_plot_1.png'):
                pix = QPixmap('img/shap_force_plot_1.png')
                self.ui.img.setPixmap(pix)
                self.ui.img.setScaledContents(True)
            else:
                print("error")

            if self.ui.LiverDisease.displayText() == "YES":
                msgBox = QMessageBox()
                msgBox.setWindowTitle('Result')
                msgBox.setText("The Patient may develop PPH!")
                msgBox.exec()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    stats = Stats()
    stats.ui.show()
    app.exec_()