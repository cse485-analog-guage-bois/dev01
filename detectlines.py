# Python program to illustrate HoughLine  
import cv2 
import numpy as np 
  
# Reading the required image in  
# which operations are to be done.  
# Make sure that the image is in the same  
# directory in which this python program is 
img = cv2.imread('line.png') 
  
# Convert the img to grayscale 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
  
# Apply edge detection method on the image 
edges = cv2.Canny(gray,300,800,apertureSize = 3) 
  

maxLineGap = 20
minLineLength = 70 
lines = cv2.HoughLinesP(edges,1,np.pi/180, 40, maxLineGap, minLineLength) 
  

if lines is not None:
    for line in lines[0]: 
        x1,y1,x2,y2 = line[0]
        cv2.line(img,(x1,y1), (x2,y2), (0,255,0),2) 
      

cv2.imwrite('linesDetected.jpg', img) 
cv2.imwrite('linesDetectedblur.jpg', edges) 
