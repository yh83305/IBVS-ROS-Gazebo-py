#!/usr/bin/env python3

import cv2
import numpy as np

def identify_key_points(centers):
    A = sorted(centers, key=lambda x: x[0])[:2]
    C = sorted(centers, key=lambda x: x[0], reverse=True)[:2]
    B = sorted(centers, key=lambda x: x[1])[:2]
    D = sorted(centers, key=lambda x: x[1], reverse=True)[:2]

    point1 = list(set(A) & set(B))
    point4 = list(set(C) & set(D))
    point3 = list(set(A) & set(D))
    point2 = list(set(B) & set(C))

    if len(point1) == 0: point1 = None
    if len(point4) == 0: point4 = None
    if len(point3) == 0: point3 = None
    if len(point2) == 0: point2 = None

    return point1, point2, point3, point4

image = cv2.imread('feature5.png')

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

lower_red = np.array([170, 120, 70])
upper_red = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv_image, lower_red, upper_red)

red_mask = cv2.bitwise_or(mask1, mask2)

kernel = np.ones((5, 5), np.uint8)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)

centers = []
for i in range(min(4, len(contours))):
    contour = contours[i]
    
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append((cX, cY))

identified_points = identify_key_points(centers)
labels = ['A', 'B', 'C', 'D']

for i, identified_point in enumerate(identified_points):
    print(identified_point)
    
    cv2.circle(image, identified_point[0], 1, (0, 255, 0), -1)
    
    cv2.putText(image, labels[i], (identified_point[0][0] + 10, identified_point[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    text = "(%d, %d)" % (identified_point[0][0], identified_point[0][1])

    cv2.putText(image, text, (identified_point[0][0] + 10, identified_point[0][1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


cv2.imshow("Detected Red Dots", image)
print(identified_points)
cv2.waitKey(0)
cv2.destroyAllWindows()


