# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 13:40:53 2025

@author: julio
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
import cv2
import numpy as np
import os




# Set the GUI skeleton using QtDesign
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1760, 986)
        MainWindow.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.logo_path = icon_path = os.path.join(os.path.dirname(__file__), "uktlogo.png")
        MainWindow.setWindowIcon(QtGui.QIcon(self.logo_path))
        self.MainWindow = MainWindow
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.OriginalVideo_frame = QtWidgets.QFrame(self.centralwidget)
        self.OriginalVideo_frame.setGeometry(QtCore.QRect(20, 50, 850, 631))
        self.OriginalVideo_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.OriginalVideo_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.OriginalVideo_frame.setObjectName("OriginalVideo_frame")
        self.OriginalVideo_label = QtWidgets.QLabel(self.OriginalVideo_frame)
        self.OriginalVideo_label.setGeometry(QtCore.QRect(0, 0, 850, 631))
        self.OriginalVideo_label.setScaledContents(True)
        self.SummedVideo_frame = QtWidgets.QFrame(self.centralwidget)
        self.SummedVideo_frame.setGeometry(QtCore.QRect(900, 50, 850, 631))
        self.SummedVideo_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.SummedVideo_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.SummedVideo_frame.setObjectName("SummedVideo_Frame")
        self.SummedVideo_label = QtWidgets.QLabel(self.SummedVideo_frame)
        self.SummedVideo_label.setGeometry(QtCore.QRect(0, 0, 850, 631))
        self.SummedVideo_label.setScaledContents(True)
        self.OriginalVideoTitle_label = QtWidgets.QLabel(self.centralwidget)
        self.OriginalVideoTitle_label.setGeometry(QtCore.QRect(20, 20, 100, 20))
        self.OriginalVideoTitle_label.setObjectName("OriginalVideoTitle_label")
        self.ReconstructedVideoTitle_label = QtWidgets.QLabel(self.centralwidget)
        self.ReconstructedVideoTitle_label.setGeometry(QtCore.QRect(900, 20, 120, 20))
        self.ReconstructedVideoTitle_label.setObjectName("ReconstructedVideoTitle_label")
        self.LoadButton_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.LoadButton_pushButton.setGeometry(QtCore.QRect(20, 700, 75, 25))
        self.LoadButton_pushButton.setObjectName("LoadButton_pushButton")
        self.LoadButton_pushButton.clicked.connect(self.upload_video)
        self.PauseButton_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.PauseButton_pushButton.setGeometry(QtCore.QRect(100, 700, 75, 25))  # Adjust position near Save button
        self.PauseButton_pushButton.setObjectName("PauseButton_pushButton")
        self.PauseButton_pushButton.setText("Pause")
        self.PauseButton_pushButton.clicked.connect(self.toggle_pause)
        self.ReplayButton_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.ReplayButton_pushButton.setGeometry(QtCore.QRect(180, 700, 75, 25))
        self.ReplayButton_pushButton.setObjectName("ReplayButton_pushButton")
        self.ReplayButton_pushButton.setText("Replay")
        self.ReplayButton_pushButton.clicked.connect(self.replay_video)
        self.ReplayButton_pushButton.setEnabled(False)
        self.SaveButton_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.SaveButton_pushButton.setGeometry(QtCore.QRect(260, 700, 75, 25))
        self.SaveButton_pushButton.setObjectName("SaveButton_pushButton")
        self.SaveButton_pushButton.clicked.connect(self.save_video)
        self.SaveButton_pushButton.setEnabled(False)
        self.ExitButton_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.ExitButton_pushButton.setGeometry(QtCore.QRect(20, 910, 75, 25))
        self.ExitButton_pushButton.setObjectName("ExitButton_pushButton")
        self.ExitButton_pushButton.clicked.connect(MainWindow.close)
        self.ShowFramesNumber_checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.ShowFramesNumber_checkBox.setGeometry(QtCore.QRect(900, 700, 160, 20))
        self.SaveLastFrame_radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.SaveLastFrame_radioButton.setGeometry(QtCore.QRect(340, 700, 160, 20))
        self.SaveLastFrame_radioButton.setObjectName("SaveLastFrame_radioButton")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1760, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)
        self.video_path = None
        self.cap = None
        self.frames = []
        self.summed_frames = []
        self.frame_index = 0
        self.fps = 30
        self.fourcc = None
        
        color=[186,168,116]
        a=0.95
        newColor = tuple (x + (1 - x) * (1 - a) for x in color)
        self.centralwidget.setStyleSheet("background-color: rgb(%i,%i,%i)" %(newColor[0],
                                                                newColor[1],
                                                                newColor[2]))
        self.UKTlabel =   QtWidgets.QLabel(self.centralwidget)
        self.pixmap = QtGui.QPixmap(self.logo_path).scaledToWidth(325)
        self.UKTlabel.setPixmap(self.pixmap)
        self.UKTlabel.setPixmap(self.pixmap)
        self.UKTlabel.resize(self.pixmap.width(),
                          self.pixmap.height())
        self.UKTlabel.move(1375,900)
        
        self.ProgressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.ProgressBar.setGeometry(QtCore.QRect(900, 725, 850, 20))  # Right below the Reconstructed title
        self.ProgressBar.setObjectName("ProgressBar")
        self.ProgressBar.setValue(0)
        self.ProgressBar.setVisible(False)
        self.paused = False

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Reconstruct autoluorescence videos"))
        self.OriginalVideoTitle_label.setText(_translate("MainWindow", "Original Video"))
        self.ReconstructedVideoTitle_label.setText(_translate("MainWindow", "Reconstructed Video"))
        self.LoadButton_pushButton.setText(_translate("MainWindow", "Load"))
        self.PauseButton_pushButton.setText(_translate("MainWindow", "Pause"))
        self.ReplayButton_pushButton.setText(_translate("MainWindow", "Replay"))
        self.SaveButton_pushButton.setText(_translate("MainWindow", "Save"))
        self.ExitButton_pushButton.setText(_translate("MainWindow", "Exit"))
        self.ShowFramesNumber_checkBox.setText(_translate("MainWindow", "Show Frame number"))
        self.SaveLastFrame_radioButton.setText(_translate("MainWindow", "Save Frame Sum"))
        
    # This method upload the video
    def upload_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.MainWindow, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")

        if not file_path:
            return

        if not file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            QtWidgets.QMessageBox.critical(self, "Invalid File", "Please select a valid video file.")
            return

        self.video_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to open the video.")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))

        self.frames = []
        self.summed_frames = []
        sum_frame = None
        

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames.append(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if sum_frame is None:
                sum_frame = np.zeros_like(gray, dtype=np.float32)

            sum_frame += gray.astype(np.float32)
            display_frame = cv2.normalize(sum_frame, None, 0, 255, cv2.NORM_MINMAX)
            display_frame = display_frame.astype(np.uint8)
            self.summed_frames.append(display_frame)     

        self.cap.release()
        self.frame_index = 0
        
        self.ReplayButton_pushButton.setEnabled(False)
        self.ProgressBar.setValue(0)
        
        self.SaveButton_pushButton.setEnabled(True)
        self.timer.start(int(1000 / self.fps))

    # This method processes the selected videos to show
    def play_video(self):
        if self.frame_index >= len(self.frames):
            self.timer.stop()
            self.ReplayButton_pushButton.setEnabled(True)
            return
        
        original = self.frames[self.frame_index]
        summed = self.summed_frames[self.frame_index].copy()
        
        if len(summed.shape) == 2 or summed.shape[2] == 1:
            summed = cv2.cvtColor(summed, cv2.COLOR_GRAY2BGR)
        
        if self.ShowFramesNumber_checkBox.isChecked():
            cv2.putText(
                summed,
                f"Frame: {self.frame_index + 1}",
                (10, 30),  # Position
                cv2.FONT_HERSHEY_SIMPLEX,  # Font
                1,  # Font scale
                (255, 255, 255),  
                2,  # Thickness
                cv2.LINE_AA
                )
            
        self.display_frame(original, self.OriginalVideo_label)
        self.display_frame(summed, self.SummedVideo_label)
        
        self.ProgressBar.setVisible(True)
        progress = int((self.frame_index / len(self.frames)) * 100)
        self.ProgressBar.setValue(progress)
        
        self.frame_index += 1
        
    # This method runs both videos after the original video is selected
    def display_frame(self, frame, label):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(label.width(), label.height()))
    
    # This method pauses the video
    def toggle_pause(self):
        if self.paused:
            # Resume the video
            self.timer.start(int(1000 / self.fps))
            self.PauseButton_pushButton.setText("Pause")
        else:
            # Pause the video
            self.timer.stop()
            self.PauseButton_pushButton.setText("Resume")
        
        # Toggle the paused state
        self.paused = not self.paused
       
    # This method replays the video
    def replay_video(self):
        self.frame_index = 0
        self.ReplayButton_pushButton.setEnabled(False)
        self.ProgressBar.setValue(0)
        
        if self.paused:
            self.paused = False
            self.PauseButton_pushButton.setText("Pause")
        
        self.timer.start(int(1000 / self.fps))
            
    # This method perform video saving 
    def save_video(self):
        if not self.video_path or not self.summed_frames:
            return
    
        base_name = os.path.basename(self.video_path)
        name, ext = os.path.splitext(base_name)
        dir_path = os.path.dirname(self.video_path)
    
        # Save the full summed video
        output_path = os.path.join(dir_path, f"{name}_sum{ext}")
        height, width, _ = self.summed_frames[0].shape
        out = cv2.VideoWriter(output_path, self.fourcc, self.fps, (width, height))
        
        for frame in self.summed_frames:
            out.write(frame)
        out.release()
    
        # If radio button checked, save last frame as .tiff
        if self.SaveLastFrame_radioButton.isChecked():
            last_frame = self.summed_frames[-1]
            tiff_path = os.path.join(dir_path, f"{name}_sum.tiff")
            cv2.imwrite(tiff_path, last_frame)
    
            QtWidgets.QMessageBox.information(
                self.MainWindow,
                "Saved",
                f"Summed video saved as:\n{output_path}\n\nLast frame saved as TIFF:\n{tiff_path}"
            )
        else:
            QtWidgets.QMessageBox.information(
                self.MainWindow,
                "Saved",
                f"Summed video saved as:\n{output_path}"
            )

        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.setWindowIcon(QtGui.QIcon(r"C:\Users\julio\LaserSafety\uktlogo.png"))
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())