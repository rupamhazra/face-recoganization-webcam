from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout,QLabel,QLineEdit
from PyQt5.QtGui import QPixmap,QIcon
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import os
from PIL import Image
import numpy as np
import pickle


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    print('change_pixmap_signal',change_pixmap_signal)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        print('continus..')
        # capture from web cam
        face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ret:
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                for (x,y,w,h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    color = (255,255,0) #BGR
                    stroke = 2
                    end_cord_x = x + w
                    end_cord_y = y + h
                    cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y), color, stroke)
                self.change_pixmap_signal.emit(frame)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    

    def __init__(self):
        super().__init__()
        self.title = 'Face Recoganization System'
        self.left = 10
        self.top = 10
        self.width = 1000
        self.height = 600
        self.setMaximumSize(self.width, self.height) 
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.createGridLayout()
        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.horizontalGroupBox)
        self.setLayout(windowLayout)
        self.show()

    def createGridLayout(self):
        self.horizontalGroupBox = QGroupBox()
        layout = QGridLayout()
        #layout.setColumnStretch(1, 4)
        self.disply_width = 650
        self.display_height = 500
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        layout.addWidget(self.image_label,0,0)

        
        layout.addWidget(self.createExampleGroup(), 0, 1)

        # self.l1 = QLabel()
        # self.l1.setText("Name")
        # layout.addWidget(self.l1,0,1)




        # self.start_bt = QPushButton('Start')
        # self.start_bt.clicked.connect(lambda:self.capture_photos(self.start_bt))
        # layout.addWidget(self.start_bt,0,1)
        print('self.update_image',self.update_image)
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

        # layout.addWidget(QPushButton('3'),0,2)
        # layout.addWidget(QPushButton('4'),0,3)
        # layout.addWidget(QPushButton('5'),0,4)
        
        self.horizontalGroupBox.setLayout(layout)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def capture_photos(self,b):
        print ("clicked button is "+b.text())
        face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        photo_id = 0
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            photo_id = photo_id + 1

            if photo_id < 100 :
                print('photo_id',photo_id)
                color = (255,255,0) #BGR
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y), color, stroke)
                img_item = "image-"+str(photo_id)+".png"
                print('img_item',img_item)
                cv2.imwrite('images/Rupam Hazra/'+img_item,roi_color)
                #cv2.imshow('frame',frame)

    def createExampleGroup(self):
        groupBox = QGroupBox()
        l_name = QLabel("Name")
        t_name = QLineEdit()
        vbox = QVBoxLayout()
        vbox.addWidget(l_name)
        vbox.addWidget(t_name)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)
        return groupBox

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())