import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

labels = {"person_name":1}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #recoginze?
        id_, conf = recognizer.predict(roi_gray)
        #print(id_,conf)
        if conf >=45:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y),font,1,color,stroke, cv2.LINE_AA)

        img_item = "my-image.png"

        cv2.imwrite(img_item,roi_color)

        color = (255,255,0) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y), color, stroke)
        subitems = smile_cascade.detectMultiScale(roi_gray)
        #print('eyes',eyes)
        for (ex,ey,ew,eh) in subitems:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),1)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()