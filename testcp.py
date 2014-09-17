#!/usr/bin/python
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    ret,thresh = cv2.threshold(gray,127,255,0)
    #cv2.imshow('frame',thresh)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = gray.shape
    blank_image = np.zeros((height,width,3), np.uint8)
    blank_image[:,:] = (0,127,0)
    cv2.drawContours(blank_image, contours, -1, (0,255,0), 1)

    # Display the resulting frame
    cv2.imshow('frame',blank_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
