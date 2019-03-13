import cv2 
import numpy as np


cimg = cv2.imread('gauge3.png') 
  
cimg = cv2.medianBlur(cimg,5)

img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img, 50, 150, apertureSize = 3)

binary_threshold = 40

#img3 = cv2.medianBlur(img,17)

#ret1, img2 = cv2.threshold(img3, binary_threshold, 255, cv2.THRESH_BINARY)
    
#edges = cv2.Canny(img2, 30, 150, apertureSize = 3)
        
lines1 = cv2.HoughLinesP(edges,1,np.pi/180, 40,maxLineGap = 20, minLineLength = 70)

if lines1 is not None:
    for line in lines1:
        x1,y1,x2,y2 = line[0]
        cv2.line(cimg,(x1,y1),(x2,y2),(0,255,0),2)
        
cv2.imwrite('edges.jpg',edges)
#cv2.imwrite('img2.jpg',img2)
cv2.imwrite('linedetect.jpg',cimg)


