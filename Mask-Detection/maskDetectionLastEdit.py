# Konoha Langsoi computer vision 2022
# 63070044 Tangkaboee Satitkasemsan
# 63070158 Visal Suwanarat
# 63070165 Sarawut Unarat 
from turtle import color
import cv2
from cv2 import VideoCapture
from cv2 import COLOR_BGR2RGB
import tensorflow
import keras
from PIL import Image, ImageOps
from keras.models import load_model
import dlib
import os
import time
# import tensorflow.keras
import numpy as np

# If you want to create dataset with your own pls callfn createDataset()
# If you want to mask detect pls callfn maskDetection()


def maskdetection():
    face_cascade = "haarcascade_frontalface_default.xml"
    # cap = cv2.VideoCapture('4K Video Stock _ Long Shot of People In Face Masks Walking Down a Busy Street In Oxford England.mp4')
    webcam = cv2.VideoCapture(0)
    countNonMask = 0
    face_classifier = cv2.CascadeClassifier(face_cascade)

    np.set_printoptions(suppress=True)

    model = tensorflow.keras.models.load_model('keras_model.h5')
    size = (224, 224)

    # face landmark
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # face count
    testCount = 0
    state = 0
    file = "UR83J6X-high-tech-computer-noises-04.mp3"
    os.system("afplay " + file)
    stat = ""
    counting_person = 0
    nonmaskCount = 0
    countMask = 0
    while True:
        if cv2.waitKey(25) & 0XFF == ord('q'):
            cv2.destroyAllWindows()
            break
        success, image_bgr = webcam.read()
        image_org = image_bgr.copy()
        image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
        faces = face_classifier.detectMultiScale(image_bw)
        facesL = detector(image_bw)
        cv2.line(image_bgr, (650, 40), (650, 650), color=(255, 60, 0), thickness=2)
        # line = cv2.line(image_bgr, (650, 40), (650, 650), color=(255, 60, 0), thickness=5)
        for face in facesL:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            landmarks = predictor(image_bw, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image_bgr, (x, y), 4, (255, 255, 255), -1)
        # cv2.line(image_bgr, (0, 120), (120, 120), color=(255, 60, 0), thickness=5)
        for face in faces:
            x, y, w, h = face
            cface_rgb = Image.fromarray(image_rgb[y:y+h,x:x+w])

            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            image = cface_rgb
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            image_array = np.asarray(image)

            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array

            prediction = model.predict(data)
            #print(prediction)

            horizontal = int(x+(w/2))
            vertical = int(y+(h/2))

            #print(w)
            #print(x)

            if prediction[0][0] > prediction[0][1]:
                cv2.putText(image_bgr,f'Masked { int(prediction[0][0] * 100) } %',(x,y-7),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,255,0),2)
                cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(image_bgr, (horizontal, vertical), radius=5, color=(255, 60, 0), thickness=-1)
                if((horizontal < (650 + 40)) and (horizontal > (650 - 40)) and (state == 1)):
                    testCount = 0
                    counting_person += 1
                    state = 0
                    stat = 'Ready'
                    countMask += 1
                    # 0 คือ not ready
                    # 1 8nv ready
                else:
                    stat = "not ready"
                
            else:
                cv2.putText(image_bgr,f'Non-Masked { int(prediction[0][1] * 100) } %',(x,y-7),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,255),2)
                cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0,0,255), 2)
                cv2.circle(image_bgr, (horizontal, vertical), radius=5, color=(255, 60, 0), thickness=-1)
                if((horizontal < (650 + 40)) and (horizontal > (650 - 40)) and (state == 1)):
                    testCount = 0
                    counting_person += 1
                    state = 0
                    stat = 'Ready'
                    countNonMask += 1
                    
                    # 0 คือ not ready
                    # 1 8nv ready
                else:
                    stat = "not ready"
            cv2.circle(image_bgr, (690, vertical), radius=5, color=(0, 255, 255), thickness=-1)
            cv2.circle(image_bgr, (610, vertical), radius=5, color=(0, 255, 255), thickness=-1)
            #เช็คว่าจุดอยู่ในช่วงใหม
            testCount += 1
            if(testCount == 30):
                state = 1
                # stat = "ready"
                # cv2.putText(image_bgr,f'Ready', (x+30,y),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,255,0),2)
            
            #print(state)
            #print(horizontal)
            print(counting_person)
        #print('xxxxx')
        #print(counting_person)
        #print(horizontal)
        #print("65")
        #วาดเส้นเเนวตั้ง
        cv2.putText(image_bgr,f'Counting : { counting_person }', (20,40),cv2.FONT_HERSHEY_TRIPLEX,1,(0,255,0),2)
        cv2.putText(image_bgr,f'non-mask : { countNonMask }', (20,90),cv2.FONT_HERSHEY_TRIPLEX, 1,(0, 0, 255),2)
        cv2.putText(image_bgr,f'mask : { countMask }', (20,140),cv2.FONT_HERSHEY_TRIPLEX,1,(0, 255, 0),2)
        cv2.putText(image_bgr,f'Status : ' + stat, (20,190),cv2.FONT_HERSHEY_TRIPLEX,1,(255, 0, 0),2)
        cv2.imshow("Mask Detection", image_bgr)
        cv2.waitKey(1)
maskdetection()
# createDataset()

