import numpy as np 
import cv2
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(r'C:\Users\Ranjit\Desktop\dl_program\face_recog_lbph\haarcascade_frontalface_default.xml')

names = ['Ranjit','Priyank']


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\Ranjit\Desktop\dl_program\face_recog_lbph\face_recog_trained_model.yml')
while True:
    _,frame=cap.read()
#test_img = cv2.imread(r'C:\Users\Ranjit\Desktop\dl_program\face_recog_lbph\dataset\1\41.jpg')
    faces=face_cascade.detectMultiScale(frame,1.32,5)
    if len(faces) == 0:
        cv2.putText(frame,'Face Not Detected',(20,30),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,0,255),3)
    else:
        for face in faces:
            x,y,w,h=face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),3)
            roi = frame[y:y+h,x:x+w]
            gray_img  = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            label,confidence = face_recognizer.predict(gray_img)
            cv2.rectangle(frame,(x,y+h),(x+w,y+h+40),(255,0,255),-1)
            cv2.putText(frame,names[label].upper(),(x+3,y+h+30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            print(label,confidence)
    cv2.imshow('image',frame)
    k = cv2.waitKey(10)
    if k == 13:
        break
cap.release()
cv2.destroyAllWindows()