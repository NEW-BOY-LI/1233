import os
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
import sys
import cv2
from matplotlib import image, widgets
from matplotlib.widgets import Widget
import mediapipe as mp
from Ui_3 import Ui_Form1  
from Ui_4 import Ui_Form
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class firstui(QWidget):
    def __init__(self, parent=None):
        super(firstui, self).__init__(parent)
        self.ui_3 = Ui_Form1()  
        self.ui_3.setupUi(self)
        self.ui_3.pushButton.clicked.connect(self.nj1)
        self.ui_3.pushButton_2.clicked.connect(self.nj2)
    
    def nj1(self):
        self.hide()
        self.s = secondui()
        self.s.show()
    
    def nj2(self):
        self.closeWindow()

    def closeWindow(self):
        self.close()

class secondui(QWidget):
    def __init__(self, parent=None):
        super(secondui, self).__init__(parent)
        self.ui_4 = Ui_Form()  
        self.ui_4.setupUi(self)
       
        self.ui_4.pushButton.clicked.connect(self.dkwj)
        self.ui_4.pushButton_2.clicked.connect(self.gb)
        self.ui_4.pushButton_3.clicked.connect(self.kscx)
        
        
        
        # 初始化 MediaPipe Pose 模型
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 定义动作判断的阈值
        self.standing_threshold = 0.1
        self.walking_threshold = 0.1
        self.sitting_threshold = 0.1
        self.hands_up_threshold = 0.1
        self.left_hand_up_threshold = 0.1
        self.right_hand_up_threshold = 0.1
        self.hands_on_hips_threshold = 0.1



    def dkwj(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "选择照片", "", "照片文件 (*.jpg *.png *.mp4)",)

        self.ui_4.lineEdit.setText(self.image_path)
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.ui_4.label.setPixmap(pixmap)
            self.ui_4.label.setScaledContents(True)

    def gb(self):
        self.closeWindow()

    def kscx(self):
        
        self.cap = cv2.VideoCapture(self.image_path)             

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to capture frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                left_ankle = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y
                right_ankle = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y
                left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y
                right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
                left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
                right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y

                standing_height_diff = abs(left_ankle - right_ankle)
                walking_height_diff = abs(left_ankle - right_ankle)
                sitting_height_diff = abs(left_ankle - right_ankle)
                hands_up_diff = abs(left_wrist - right_wrist)
                left_hand_up_diff = abs(left_wrist - left_shoulder)
                right_hand_up_diff = abs(right_wrist - right_shoulder)
                hands_on_hips_diff = abs(left_wrist - right_wrist)

                if standing_height_diff < self.standing_threshold:
                    cv2.putText(frame, "Standing", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                elif walking_height_diff > self.walking_threshold:
                    cv2.putText(frame, "Walking", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                elif sitting_height_diff < self.sitting_threshold:
                    cv2.putText(frame, "Sitting", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                elif hands_up_diff > self.hands_up_threshold:
                    cv2.putText(frame, "Hands Up", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                elif left_hand_up_diff > self.left_hand_up_threshold:
                    cv2.putText(frame, "Left Hand Up", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                elif right_hand_up_diff > self.right_hand_up_threshold:
                    cv2.putText(frame, "Right Hand Up", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                elif hands_on_hips_diff < self.hands_on_hips_threshold:
                    cv2.putText(frame, "Hands on Hips", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            #cv2.imshow('BlazePose Action Detection', frame)
            # 将 OpenCV 图像转换为 QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # 将 QImage 转换为 QPixmap
            pixmap = QPixmap.fromImage(q_image)

            # 将 QPixmap 设置到 label 上
            self.ui_4.label.setPixmap(pixmap)
            self.ui_4.label.setAlignment(Qt.AlignCenter)

            # 刷新界面
            QtWidgets.QApplication.processEvents()

            pixmap = QPixmap.fromImage(q_image)
            self.ui_4.label.setPixmap(pixmap)
            self.ui_4.label.setScaledContents(True)


        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWnd = firstui()
    myWnd.show()  
    sys.exit(app.exec_())
