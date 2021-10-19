import numpy as np
import cv2
import os
import face_recognition as fr
from face_recognition.api import face_locations
from datetime import datetime

path = r"C:\Users\KIIT\Desktop\Apps\Face_R_GIT\face_recognition\Face Detection & Attendence\Images"
images=[]
face_cascade= cv2.CascadeClassifier(r"C:\Users\KIIT\Desktop\Apps\Face_R_GIT\face_recognition\Face Detection & Attendence\haarcascade_frontalface_default.xml")

classNames= []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg= cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

def findEncoding(images):
    encodeList=[]
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode= fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def MarkAttendence(images):
    with open(r"C:\Users\KIIT\Desktop\Apps\Face_R_GIT\face_recognition\Face Detection & Attendence\Attendence_Report.csv",'r+')as f:
        myDataList= f.readlines()
        nameList=[]
        for line in myDataList:
            entry= line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now= datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

# MarkAttendence('Elon')

encodeListKnown= findEncoding(images)
print('Encoding Complete')

cap= cv2.VideoCapture(0)
while cap.isOpened():
    _,img= cap.read()
    imgS= cv2.resize(img,(0,0),None,0.25,0.25)
    imgS= cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    faces= face_cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)

    faceLocCurr= fr.face_locations(imgS)
    encodeCurr= fr.face_encodings(imgS,faceLocCurr)

    for encodeFace,faceLoc in zip(encodeCurr,faceLocCurr):
        matches = fr.compare_faces(encodeListKnown,encodeFace)
        faceDis= fr.face_distance(encodeListKnown,encodeFace)
        # print(faceDis)
        matchesIndex = np.argmin(faceDis)

        if matches[matchesIndex]:
            name= classNames[matchesIndex].upper()
            print(name)
            y1,x2,y2,x1= faceLoc
            # print(faceLoc)
            y1,x2,y2,x1= y1*4,x2*4,y2*4,x1*4
            x,y,w,h=y1*4,x2*4,y2*4,x1*4
            
            # print(x,y,w,h)
            MarkAttendence(name)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            # cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) 

            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            # cv2.imshow

            # cv2.imshow('webcam',img)
            # cv2.waitKey(1)
    cv2.imshow('Web Cam',img)
    if cv2.waitKey(1)==27:
        break

cap.release()
    
cv2.destroyAllWindows()





# faceLoc= fr.face_locations(imgElon)[0]
# encodeElon= fr.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) 

# faceLocTest= fr.face_locations(imgTest)[0]
# encodeElonTest= fr.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2) 

# results = fr.compare_faces([encodeElon],encodeElonTest)
# faceDis= fr.face_distance([encodeElon],encodeElonTest)

