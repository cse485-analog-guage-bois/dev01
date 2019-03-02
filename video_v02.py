import cv2 
import numpy as np
import imutils
from imutils.video import VideoStream

#img = cv2.imread("gauge-7.jpg") # Read jpg image
print ("[INFO] Starting video stream...")
video = cv2.VideoCapture(0)

while True:
    check, cimg = video.read()
    
    cimg = cv2.medianBlur(cimg,5)
    img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 500, param1=250,param2=20,minRadius=10,maxRadius=80)
    
    if circles is None:
        continue
        
    for i in circles[0,:]:
        cv2.circle(cimg,(i[0],i[1]),i[2],(255,0,0),2)

    binary_threshold = 40
    img3 = cv2.medianBlur(img,17)
    ret1, img2 = cv2.threshold(img3, binary_threshold, 255, cv2.THRESH_BINARY)
    
    edges = cv2.Canny(img2, 30, 150, apertureSize = 3)
        
    lines1 = cv2.HoughLinesP(edges,1,np.pi/180, 40,maxLineGap = 20, minLineLength = 70)

    if lines1 is not None:
        for line in lines1:
            x1,y1,x2,y2 = line[0]
            cv2.line(cimg,(x1,y1),(x2,y2),(0,255,0),2)
        
    cv2.imshow('edges',edges)
    cv2.imshow('img2',img2)
    cv2.imshow('video',cimg)
    
    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()

