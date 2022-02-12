# to import libraries
import cv2  # for computer vision functions
import numpy as np  # scientific computing library
import math  # maths functions library

# to open video
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # to set width
cap.set(4, 480)  # to set height


# function that returns nothing
def empty(a):
    pass


'''used the above function for trackbars to set HSV values'''


# function to calculate angle
def getangle(x1, y1, x2, y2):
    slope = (x2 - x1) / (y2 - y1)  # inverse of slope for angle with vertical axis(y-axis)
    theta = math.atan(slope)  # tan inverse function
    theta = theta * 57.3  # to convert radians into degrees

    if theta < 0:
        return theta + 360  # to keep range of angle from 0 to 360
    else:
        return theta


# function to find contours
def getcontours(img, imgcontour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # to find contours in input image

    c = max(contours, key=cv2.contourArea)  # to find contour with max area

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1000:  # to only detect contours with area>1000
            cv2.drawContours(imgcontour, cnt, -1, (0, 0, 255), 5)  # to draw red contours
            peri = cv2.arcLength(cnt, True)  # epsilon value to feed in approxPolyDP
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # finds closed contour with approx polygonal shape
            print(len(approx))  # to print number of vertices in shape

            if len(approx) == 7:  # arrow has 7 vertices, so this should work only if it detects an arrow

                x, y, w, h = cv2.boundingRect(approx)  # to find parameters of rectangle around arrow
                cv2.rectangle(imgcontour, (x, y), (x + w, y + h), (0, 255, 0), 2)  # to draw rectangle around arrow

                cv2.drawContours(imgcontour, [c], -1, (36, 255, 12), 2)  # to draw contours around arrow

                n = approx.ravel()  # to store coordinates of vertices of arrow
                x1 = n[0]  # x-coordinate of head
                y1 = n[1]  # y-coordinate of head
                x2 = (n[6] + n[8]) / 2  # x-coordinate of tail
                y2 = (n[7] + n[9]) / 2  # y-coordinate of tail

                angle = getangle(x1, y1, x2, y2)  # calculating angle using coordinates

                # to put text on image about direction of arrow with vertical
                cv2.putText(imgcontour, 'Direction: ' + str(angle), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)


# loop to run video because video is a loop of images
while True:

    success, img = cap.read()  # to input camera feed
    imgcontour = img.copy()  # copying original image to draw contours

    # code to separate red colour only

    imgblur = cv2.GaussianBlur(img, (9, 9), 1)  # blur image of original image
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray image of original image
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV image of original BGR image

    # setting mask2 HSV values
    h_min2 = 0
    h_max2 = 179
    s_min2 = 0
    s_max2 = 255
    v_min2 = 0
    v_max2 = 0
    lower2 = np.array([h_min2, s_min2, v_min2])
    upper2 = np.array([h_max2, s_max2, v_max2])

    # setting mask1 HSV values
    h_min1 = 0
    h_max1 = 179
    s_min1 = 155
    s_max1 = 255
    v_min1 = 170
    v_max1 = 255
    lower1 = np.array([h_min1, s_min1, v_min1])
    upper1 = np.array([h_max1, s_max1, v_max1])

    mask1 = cv2.inRange(imghsv, lower1, upper1)
    mask2 = cv2.inRange(imghsv, lower2, upper2)
    mask = mask1 + mask2  # applying both masks to a single mask

    result = cv2.bitwise_and(img, img, mask=mask)  # masking original image with masked image

    imgcanny = cv2.Canny(result, 100, 255)  # canny image of result image

    kernel2 = np.ones((5, 5))  # matrix to feed in dilated image
    imgdilation = cv2.dilate(imgcanny, kernel2, iterations=1)  # dilated image of canny image

    getcontours(imgdilation, imgcontour)  # applying contours on image

    cv2.imshow('Contour Image', imgcontour)  # to show contour video
    cv2.imshow('Original Image', img)  # to show original video

    # condition to exit code on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

