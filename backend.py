import cv2
import numpy as np
from keras.models import load_model
#To read a video and detect the people wearing mask in it.
facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model('mask.h5',compile=False)
vid=cv2.VideoCapture("group mask.mp4")
while(vid.isOpened()):
    flag,frame=vid.read()
    if (flag):
        faces=facemodel.detectMultiScale(frame)
        for(x,y,l,h) in faces:
            #Cropping the face
            face_img=frame[y:y+h,x:x+l]
            #resize the crop face
            face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
            #Converting the shape of image according to the desired dimension of model
            face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
            #Normalizing the image
            face_img=(face_img/127.5)-1
            p=maskmodel.predict(face_img)[0][0]
            if (p>0.9):
                cv2.rectangle(frame,(x,y),(x+l,y+h),(0,0,255),4)
            else:
                cv2.rectangle(frame,(x,y),(x+l,y+h),(0,255,0),4)
        cv2.namedWindow("sharbani window",cv2.WINDOW_NORMAL)
        cv2.imshow("sharbani window",frame)
        k=cv2.waitKey(1)
        if(k==ord('x')):
            break
    else:
        break
cv2.destroyAllWindows()

