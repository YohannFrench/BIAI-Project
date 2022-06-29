from calendar import c
from tkinter import Frame, font
import cv2
from cv2 import COLOR_BAYER_BG2GRAY
import matplotlib.pyplot as plt

#loading image using cv2

img = cv2.imread("happy_boy.jpg") # fonction pour  charger une image

### First step generating the raw photo
###showing image using plt in color BGR by default

#plt.imshow(img)
#plt.show()

### We transform the color BGR in RGB to have the real color of the photo
# We use cv2.cvtColor and then showing the image with plt
color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(color_img)
plt.show()

#importing deepface library and DeepFace
from deepface import DeepFace

#this analyses the given image and gives values
#when we use this for 1st time, it may give many errors and some google drive links to download some '.h5' and zip files, download and save them in the location where it shows that files are missing.
prediction = DeepFace.analyze(color_img)
prediction['dominant_emotion']



#loading our xml file into faceCascade using cv2.CascadeClassifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detecting face in color_image and getting 4 points(x,y,u,v) around face from the image, and assigning those values to 'faces' variable 
faces = faceCascade.detectMultiScale(color_img, 1.1, 4)

#using that 4 points to draw a rectangle around face in the image
for (x, y, u, v) in faces:
    cv2.rectangle(color_img, (x,y), (x+u, y+v), (0, 0, 225), 2)
    
plt.imshow(color_img)

#Afficher avec l'émotion sur l'image
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(color_img,prediction['dominant_emotion'],(0,500), font, 1, (255,0,0),2)

#Avec la caméra
# define a video capture object
cam = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = cam.read()
    prediction = DeepFace.analyze(frame, actions=['emotion'], enforce_detection= False)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(color_img, 1.1, 4)
    
    for (x, y, u, v) in faces:
        cv2.rectangle(color_img, (x,y), (x+u, y+v), (0, 0, 225), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(color_img,prediction['dominant_emotion'],(0,500), font, 1, (255,0,0),2)    
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()



