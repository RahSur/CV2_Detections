import cv2
from random import randrange

img = cv2.imread('cap.jpg')

trained_face_date = cv2.CascadeClassifier('car_detect.xml')

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

print("Code Completed")