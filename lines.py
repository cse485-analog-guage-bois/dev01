import cv2
import numpy as np
#import sys
#from imutils.video import VideoStream
#from imutils.video import FPS
#from matplotlib import pyplot as plt
#import paho.mqtt.client as mqtt
#import time


def avg_lines(lines):
  line_slopes = []
  line_weights = []
  for line in lines:
    for x1,y1,x2,y2 in line:
      if x2==x1:
        continue
      slope = (y2-y1)/(x2-x1)
      intercept = y1 - slope*x1
      length = np.sqrt((y2-y1)**2+(x2-x1)**2)
      line_slopes.append((slope,intercept))
      line_weights.append((length))
  line_avg = np.dot(line_weights,line_slopes)/(np.sum(line_weights)) if len(line_weights) > 0 else None
  return line_avg


def convert_into_pixel_points(y1,y2,line):
  if line is None:
    return None

  slope, intercept = line
  x1 = int((y1 - intercept)/slope)
  x2 = int((y2 - intercept)/slope)
  y1 = int(y1)
  y2 = int(y2)

  return ((x1,y1),(x2,y2))
  


def crop_image(img, height, width):
  img1 = cv2.imread('gauge-3.jpg')
  gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

  # Create mask
  mask = np.zeros((height,width), np.uint8)

  edges = cv2.Canny(thresh, 100, 200)
  #cv2.imshow('detected ',gray)
  #cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
  for i in circles[0,:]:
    i[2]=i[2]+4
    # Draw on mask
    cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)

  # Copy that image using that mask
  masked_data = cv2.bitwise_and(img1, img1, mask=mask)

  # Apply Threshold
  _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

  # Find Contour
  contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  x,y,w,h = cv2.boundingRect(contours[0])

  # Crop masked_data
  crop = masked_data[y:y+h,x:x+w]

  #cv2.imshow('cropped image', crop)
  
  return crop


def calibration(img, h, w):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,9)
    edges = cv2.Canny(blur,75,100)

    #blur = cv2.medianBlur(gray,57)
    #edges = cv2.Canny(blur,75,100)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.2, 100)
    if circles is not None:
      circles = np.round(circles[0,:].astype("int"))
      for (x,y,r) in circles:
        cv2.circle(img,(x,y),r,(255,0,255),4)
        cv2.rectangle(img,(x-5,y-5),(x+5,y+5),(0,255,0),-1)
        separation = 10.0  #in degrees
        interval = int(360 / separation)
        p1 = np.zeros((interval,2))  #set empty arrays
        p2 = np.zeros((interval,2))
        p_text = np.zeros((interval,2))
        for i in range(0,interval):
          for j in range(0,2):
            if (j%2==0):
              p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
            else:
              p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
        text_offset_x = 10
        text_offset_y = 5
        for i in range(0, interval):
          for j in range(0, 2):
            if (j % 2 == 0):
              p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
              p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
            else:
              p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
              p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  
              # point for text labels, i+9 rotates the labels by 90 degrees

        #add the lines and labels to the image
        for i in range(0,interval):
          cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(255, 0, 255), 2)
          cv2.putText(img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(255,0,255),1,cv2.LINE_AA)
    return


def main():
  img = cv2.imread('gauge-3.jpg')
  height,width = img.shape[:2]
  #crop = crop_image(img,height,width)
  blur = cv2.medianBlur(img,9)
  edges = cv2.Canny(blur,75,100)
  calibration(img,height,width)
  #edges_new = cv2.Canny(new,75,100)
  #lines_new = cv2.HoughLinesP(edges_new,1,np.pi/180,50,maxLineGap = 40,minLineLength = 200)
  #if lines_new is not None:
  #  for line_new in lines_new:
  #    x1,y1,x2,y2 = line_new[0]
  #    new_img = cv2.line(new,(x1,y1),(x2,y2),(0,255,0),5)
  #cv2.imshow("frame_new", new)
  #cv2.imshow("edges_new", edges_new)
  #cv2.imshow("new_img", new_img)
  #cv2.waitKey(10000)
  lines = cv2.HoughLinesP(edges,1,np.pi/180,50,maxLineGap = 40,minLineLength = 200)
  lines_avg = avg_lines(lines)
  #print("line avg")
  #print(lines_avg)
  if lines is not None:
    for line in lines:
      #print ("line")
      #print(line)
      x1,y1,x2,y2 = line[0]
      #cv2.line(img,(x1,y1),(x2,y2),(0,255,0),5)
      (x3,y3),(x4,y4) = convert_into_pixel_points(y1,y2,lines_avg)
      cv2.line(img,(x3,y3),(x4,y4),(255,0,255),3)
      cv2.rectangle(img,(x3-5,y3-5),(x3+5,y3+5),(0,255,0),-1)
      # Get angle of gauge needle
      angle = np.rad2deg(np.arctan2(y4 - y3, x4 - x3))
      # Get 360 degree equivelent value of needle angle
      #print ("ANGLE")
      #print (angle)
      location = angle + 90
      #***********************************
      # Set values for gauge type
      #min_angle = 50
      #max_angle = 310
      #min_value = 0
      #max_value = 250
      min_angle = 45.0
      max_angle = 310.0
      min_value = 0.0
      max_value = 70.0
      #************************************

      #************************************
      # Calculation
    new_max = max_angle - min_angle
    #print ("new max")
    #print (new_max)
    new_min = 0.0
    new_tick_distance = max_value / new_max
    #print ("new tick distance")
    #print (new_tick_distance)
    new_location = location - min_angle
    #print ("new location")
    #print (new_location)
    final_value = new_location * new_tick_distance
    #print ("final value")
    print (final_value)
      #************************************
  cv2.imshow("blur", blur)
  cv2.imshow("frame", img)
  cv2.imshow("edges", edges)
  #cv2.imshow('cropped image', crop)
  cv2.waitKey(10000)


if __name__=='__main__':
	main()
