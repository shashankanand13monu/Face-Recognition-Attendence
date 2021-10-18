import cv2
import numpy as np

face_cascade= cv2.CascadeClassifier('C:\\Users\\KIIT\\Desktop\\Apps\\Face_R_GIT\\face_recognition\\Attendence Project\\haarcascade_frontalface_default.xml')


# img= cv2.imread('road.jpg')
cap= cv2.VideoCapture(0)

while cap.isOpened():
    _,img=cap.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    faces= face_cascade.detectMultiScale(gray,1.1,4)
    

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)



    cv2.imshow('img',img)
    if cv2.waitKey(1)==27:
        break

cap.release()






