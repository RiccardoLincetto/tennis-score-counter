import os

import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


def find_rectangles(img: np.ndarray) -> list:
    """
    Get list of rectangles in image.

    :param img: input image.
    :return: list of rectangles in image.
    """
    rectangles = []  # output initialization

    # Find edges
    # OpenCV Canny edge detection contains already:
    # 1. Gaussian smoothing
    # 2. Derivative (Sobel kernel)
    # 3. Non-maximal suppression
    # 4. Hysteresis thresholding
    edges = cv2.Canny(frame, 50, 100)

    # Find contours
    # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    # Use CHAIN_APPROX_SIMPLE to remove redundant points of straight edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find rectangles
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        # area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # x, y, w, h = cv2.boundingRect(contour)

        # Filter rectangles
        if len(contour) == 4 and cv2.contourArea(contour) > 1000 and cv2.isContourConvex(contour):
            cnt = contour.reshape(-1, 2)
            max_cos = np.max([utils.angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4]) for i in range(4)])
            if max_cos < 0.1:
                rectangles.append(cnt)

    if __name__ == "__main__":
        # Draw edges
        cv2.imwrite(os.path.join("..", "data", "sample-edges.jpg"), edges)
        # Draw contours
        frame_contours = frame.copy()
        cv2.drawContours(frame_contours, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(os.path.join("..", "data", "sample-contours.jpg"), frame_contours)
        # Draw rectangles
        cv2.drawContours(frame, rectangles, -1, (0, 255, 0), 1)
        cv2.imwrite(os.path.join("..", "data", "sample-rects.jpg"), frame)

    return rectangles


# TODO replace with find_rectangles
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
                if len(cnt) == 4 and 1000 < cv2.contourArea(cnt) < bin.shape[0] * bin.shape[1] * 0.05 \
                        and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([utils.angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


# TODO try using L1-norm instead of L2-norm for favouring corners
def rate_rectangles(img: np.ndarray, rectangles: list) -> np.ndarray:
    """
    Score rectangles proportionally to their likelihood of being the scoreboard.

    Scoreboard is usually on the corners, so the further away from the centre of the image, the better.
    Histogram of the scoreboard tends to be flat.
    Alphanumeric characters are within it.

    :param img: input image.
    :param rectangles: list of rectangles, expressed as 4 x 2 lists containing the vertices.
    :return: scores for rectangles, of the same length of input list. Each score is a value in range [0,1],
        but the output vector does not represent a pdf yet, for visualization purposes.
    """
    rates = np.zeros(len(rectangles))  # output initialization

    # Centre distance
    img_centre = np.array(img.shape[:2]) // 2  # frame central coordinates
    max_dist = np.linalg.norm(img_centre, 1)  # maximum distance from centre (computed with [0,0] and [xc,yc])
    for i, rect in enumerate(rectangles):
        rect_centre = np.mean(rect, axis=0)
        rates[i] = np.linalg.norm(rect_centre - img_centre) / max_dist

    return rates  # TODO return pdf: / np.sum(rates)


if __name__ == "__main__":  # Single image processing

    import argparse
    from random import choice
    import utils

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", dest="debug", default=False)
    args = parser.parse_args()

    # Extract frame with available annotations
    annotations = utils.read_annotations(
        os.path.join("..", "data", "top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json"))
    k = choice(list(annotations))
    annotations = annotations[k]
    frame = utils.extract_frame(os.path.join("..", "data", "top-100-shots-rallies-2018-atp-season.mp4"), int(k))
    if args.debug:
        frame_out = frame.copy()
        # cv2.drawContours(frame_out, [], -1, (0, 255, 0), 3)  # TODO check required contour format
        cv2.imwrite(os.path.join("..", "data", "debug", "frame.png"), frame_out)
        del frame_out

    # Extract groundtruth scoreboard
    box = utils.extract_box_from_frame(frame, annotations['bbox'])
    if args.debug:
        frame_out = box.copy()
        # cv2.drawContours(frame_out, [], -1, (0, 255, 0), 3)  # TODO check required contour format
        cv2.imwrite(os.path.join("..", "data", "debug", "frame-scoreboard.png"), frame_out)
        del frame_out

    # Groundtruth OCR
    print(f"From whole image:\n{pytesseract.image_to_string(frame)}", end="\n\n")
    print(f"From annotated box:\n{pytesseract.image_to_string(box)}", end="\n\n")

    # Extract rectangles
    squares = find_squares(frame)
    # squares = find_rectangles(frame)
    if args.debug:
        frame_out = frame.copy()
        cv2.drawContours(frame_out, squares, -1, (0, 255, 0), 3)
        cv2.imwrite(os.path.join("..", "data", "debug", "frame-rects.png"), frame_out)
        del frame_out

    # Rate rectangles
    scores = rate_rectangles(frame, squares)
    scores_sorted_index = np.flip(np.argsort(scores))
    if args.debug:
        frame_out = frame.copy()
        for i, rect in enumerate(squares):
            cv2.drawContours(frame_out, [rect], -1, [int(scores[i] * 255)] * 3, 5)
        cv2.imwrite(os.path.join("..", "data", "debug", "frame-rects-scored.png"), frame_out)
        del frame_out
