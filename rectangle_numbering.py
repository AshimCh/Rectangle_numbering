import cv2
import numpy as np

# Reading, grayscale and threshold
img = cv2.imread('R3.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,50,255,1)

# Find contours 
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for i,c in enumerate(contours):
    if hierarchy[0,i,3] == -1 and cv2.contourArea(c)>50:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, str(w), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        m = cv2.moments(c)
        cx,cy = m['m10']/m['m00'],m['m01']/m['m00']
        cv2.circle(img,(int(cx),int(cy)),3,255,-1)


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
