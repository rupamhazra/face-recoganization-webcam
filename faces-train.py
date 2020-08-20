
import cv2
import os
from PIL import Image
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'images')

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = list()
x_train = list()

for root, dirs, files in os.walk(image_dir):
    #print('files',files)
    for file in files:
        if file.endswith("jfif") or file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root, file)
            #print('path',path)
            label = os.path.basename(root).replace(" ","-").lower()
            print('label',label,path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id_ = label_ids[label]
            #print(id_)
            #y_labels.append(label) # some number
            #x_train.append(path) # verify the image, turn into a NUMPY array GRAY
            pil_image = Image.open(path).convert('L') # gray 
            # Resize Image
            size = (250,250)
            final_image = pil_image.resize(size,Image.ANTIALIAS)
            image_array= np.array(final_image,"uint8") # convert image into numbers
            #print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle","wb") as f:
    pickle.dump(label_ids, f)

#print(x_train)
recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainner.yml')
