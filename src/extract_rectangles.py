import math
import os

import cv2
import numpy as np


def plot_lines(image, lines):
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


# TODO use a setup script for folder structure setup and data download
if not os.path.isdir(os.path.join("..", "data", "frames")):
    os.mkdir(os.path.join("..", "data", "frames"))

# Open video
cap = cv2.VideoCapture(os.path.join("..", "data", "top-100-shots-rallies-2018-atp-season.mp4"))
assert cap.isOpened(), print("Error opening video stream or file")

idx = 0  # frame index

while cap.isOpened():
    # Capture frame-by-frame
    print(idx)
    ret, frame = cap.read()
    if not ret:
        cap.release()

    if idx % 250 == 0:
        # cv2.imwrite(os.path.join("data", "frames", f"{idx}.png"), frame)

        # Compute edges
        # edges = cv2.Canny(cv2.GaussianBlur(frame, (5, 5), 1), 150, 100)
        # cv2.imwrite(os.path.join("data", "edges", f"{idx}.png"), edges)

        # Get lines
        # The relevant ones are perfectly horizontal and vertical
        # lines = cv2.HoughLines(edges, 1, np.pi / 360, 150, None, 0, 0)
        # plot_lines(frame, lines)

        # Draw squares
        squares = find_squares(frame)
        cv2.drawContours(frame, squares, -1, (0, 255, 0), 3)

        cv2.imwrite(os.path.join("..", "data", "frames", f"{idx}.png"), frame)

    # Update frame index
    idx += 1

cap.release()
