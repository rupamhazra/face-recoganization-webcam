from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit,QDialog,QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets,uic
from PyQt5.QtGui import QPixmap,QIcon
import sys
import os
from PIL import Image
import numpy as np
import pickle
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
import time
import pyttsx3 #voice output

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'images')

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

label_ids = {}
y_labels = list()
x_train = list()

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi('screen.ui',self)
        # Initiate text to voice
        self.engine = pyttsx3.init()
        self.capture_photo = False
        self.photo_id = 0
        self.current_id = 0
        self.detect_id = 0
        self.confirm = False
        self.face_detect_activate = False
        #self.stop_bt.clicked.connect(lambda:self.stop_cam(self.stop_bt))
        self.pushButton_4.clicked.connect(lambda:self.capture_photos())
        #self.progressBar.setValue(65)
        self.timer = QTimer()
        self.timer.timeout.connect(self.detectFaces)
        self.start_bt.clicked.connect(self.controlTimer)
        self.train_model_bt.clicked.connect(self.train_model)
        self.face_detect_bt.clicked.connect(self.face_detect)
        #self.dialog_bt.clicked.connect(self.messageBox)
        self.show()

    def detectFaces(self):
        if self.capture_photo:
            name = self.name.text().replace(' ','-')
            empid = self.empid.text().replace(' ','-')
            folder_name =  name+'-'+empid
            #print('folder_name',folder_name)
            if not os.path.exists('images/'+folder_name):
                os.makedirs('images/'+folder_name)
        
        if self.face_detect_activate:
            recognizer.read('trainner.yml')
            labels = {"person_name":1}
            with open("labels.pickle",'rb') as f:
                og_labels = pickle.load(f)
                labels = {v:k for k,v in og_labels.items()}    

        # read frame from video capture
        ret, frame = self.cap.read()

        # resize frame image
        #scaling_factor = 0.8
        #frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # convert frame to GRAY format
        #print('frame',frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect rect faces
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        

        # for all detected faces
        
        for (x, y, w, h) in face_rects:
            # draw green rect on face
            roi_color = frame[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]
            color = (255,255,0) #BGR
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            #cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y), color, stroke)
            self.draw_border(frame,(x,y),(end_cord_x,end_cord_y), color, stroke,10,20)

            if self.capture_photo:
                self.photo_id = self.photo_id + 1
                #print('photo_id',self.photo_id)
                if self.photo_id < 11 :
                    #print('photo_id',photo_id)
                    img_item = "image-"+str(self.photo_id)+".png"
                    cv2.imwrite('images/'+folder_name+'/'+img_item,roi_color)
                    
                else:
                    self.controlTimer()
                    self.messageBox('Sucessfully captured your photo')
                    
            if self.face_detect_activate:
                id_, conf = recognizer.predict(roi_gray)
                if conf >=45:
                    font = cv2.FONT_HERSHEY_PLAIN
                    name = labels[id_]
                    color = (255,255,255)
                    stroke = 2
                    cv2.putText(frame,name,(x-80, y-10),font,2,color,stroke)
                    #self.confirm = True
                    #time.sleep(2)
                    #self.confirmFace()

        # convert frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get frame infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from RGB frame
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # show frame in img_label
        self.video_label.setPixmap(QPixmap.fromImage(qImg))
        # if self.face_detect_activate:
        #     self.detect_id +=1
        #     if self.confirm:
        #         #time.sleep(5)
        #         self.confirmFace()

    def draw_border(self,img, pt1, pt2, color, thickness, r, d):
        x1,y1 = pt1
        x2,y2 = pt2

        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.start_bt.setText("Stop Webcam")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.start_bt.setText("Start Webcam")
            self.face_detect_activate = False
            self.capture_photo = False

    def capture_photos(self):
        self.capture_photo = True
    
    def train_model(self):
        #self.dialog()
        for root, dirs, files in os.walk(image_dir):
            #print('files',files)
            for file in files:
                if file.endswith('jfif') or file.endswith('png') or file.endswith('jpg'):
                    path = os.path.join(root, file)
                    #print('path',path)
                    label = os.path.basename(root).replace(" ","-").lower()
                    print('label',label,path)

                    if not label in label_ids:
                        label_ids[label] = self.current_id
                        self.current_id +=1
                    id_ = label_ids[label]
                    #print(id_)
                    #y_labels.append(label) # some number
                    #x_train.append(path) # verify the image, turn into a NUMPY array GRAY
                    pil_image = Image.open(path).convert('L') # gray 
                    # Resize Image
                    size = (450,450)
                    final_image = pil_image.resize(size,Image.ANTIALIAS)
                    image_array= np.array(final_image,"uint8") # convert image into numbers
                    #print(image_array)

                    faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=5)

                    for (x,y,w,h) in faces:
                        roi = image_array[y:y+h,x:x+w]
                        x_train.append(roi)
                        y_labels.append(id_)

        with open("labels.pickle","wb") as f:
            pickle.dump(label_ids, f)
        #print(x_train)
        recognizer.train(x_train, np.array(y_labels))
        recognizer.save('trainner.yml')
        #self.speak('Models trained successfully')
        self.messageBox('Models trained successfully')
        
    def face_detect(self):
        self.face_detect_activate = True
        
    def dialog(self):
        self.p_dialog = ProgresDialog()
        self.p_dialog.show()
        #widget.setCurrentIndex(widget.currentIndex()+1)
    
    def messageBox(self,text):
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Information)
        msgbox.setWindowTitle('Alert')
        msgbox.setText(text)
        msgbox.exec_()

    def confirmFace(self):
        self.speak('Authenticated successfully')
        self.controlTimer()
        self.messageBox('Authenticated successfully')
        #time.sleep(5)
    
    def speak(self,audio):
        self.engine.say(audio)
        self.engine.runAndWait()


class ProgresDialog(QDialog):
    def __init__(self):
        super(ProgresDialog,self).__init__()
        uic.loadUi('dialog.ui',self)


app = QApplication(sys.argv)
UIWindow = UI()
UIWindow.setMaximumHeight(558)
UIWindow.setMaximumWidth(860)
app.exec_()
