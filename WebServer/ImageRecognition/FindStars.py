import cv2
import numpy as np 
import collections
import sys
from PIL import Image
from math import sqrt

# BGR [low] [high] values OpenCV reverses color order
boundaries = [
    ([173, 63, 0], [255, 179, 135]), # Blue mask lower/upper
    ([0, 69, 200], [0, 147, 255])# Orange mask Lower/Upper        
]

#img = cv2.imread(str(sys.argv[1]),0)
#Read in image
img_color = cv2.imread('image2.png')
img = cv2.imread('image2.png', 0)

#cv2.imshow("grayscale", img)
#cv2.waitKey(0)

def countStars(image):
    ret,thresh = cv2.threshold(image,127,255,0)
    img, contours, hierarchy = cv2.findContours(thresh,1,2)
    
    
    no_of_vertices = []
    
    i = 0
    mask = np.zeros(img.shape,np.uint8)
    
    starCount = 0
    divMaxSize = 0.175
    divMinSize = 0.125
    for contour in contours:
    
        cnt = contour
        area = cv2.contourArea(cnt)
        arcLen = cv2.arcLength(cnt, True)
        prop = sqrt(area)/arcLen
        if(prop < divMaxSize and prop > divMinSize):
            #print("I'm a star")
            starCount += 1
         
    print(starCount)

countStars(img)

# loop over the boundaries
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
 
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(img_color, lower, upper)
    countStars(mask)
 
	# show the images
    #cv2.imshow("images", img_color)
    #cv2.waitKey(0)
"""if area>150:
    epsilon = 0.02*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    no_of_vertices.append(len(approx))



counter = collections.Counter(no_of_vertices)




 a,b = counter.keys(),counter.values()

 i=0
 while i<len(counter):
   print a[i],b[i]
   i = i + 1"""