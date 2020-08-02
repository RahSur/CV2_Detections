import cv2
from random import randrange

#loads some pre trained face fontals from openCV (Haar Cascade Algorithm)
trained_face_date = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an img to detect faces in 

#img = cv2.imread('face.png')
#img = cv2.imread('two.png')
#img = cv2.imread('multiple.png')

#Real time Video face detection
#0 stands or default video(webcam)
# () inside this can also be a video file '.mp4'
webcam  =  cv2.VideoCapture(0)

#need to loop through all the frames captured when real time from web cam
#successfull_frame (user defined) is a boolean to say whether read is true or false
while True:
    successfull_frame, frame = webcam.read()

    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    faces_coordinates = trained_face_date.detectMultiScale(grayscale_img)

    for (x,y,w,h) in faces_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256) ,randrange(256) , randrange(256)), 4)

        cv2.imshow('Rahul Face Detector', frame)
        #waitkey() means it waits infinitely until a key pressed.
        #(1) mean is waits for 1 ms and moves on to next frame 
        key = cv2.waitKey(1) 

        if key==81 or key==113:
            break

webcam.release()    



'''
#must convert to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
faces_coordinates = trained_face_date.detectMultiScale(grayscale_img)

for (x,y,w,h) in faces_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(256) ,randrange(256) , randrange(256)), 4)


print(faces_coordinates)

#the pic pops up with a name in quotes
cv2.imshow('Rahul Face Detector', img)

#the pic waits until some key is pressed
cv2.waitKey()
'''

print("Face Detection using OpenCV")