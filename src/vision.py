import os
import pathlib

import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
OUTPUT_PATH = pathlib.Path("../data/debug")


def find_rectangles(img: np.ndarray) -> list:
    """
    Get list of rectangles in image.

    :param img: input image.
    :return: list of rectangles in image.
    """
    rectangles = []  # output initialization

    # Canny edge detection does this already, but not threshold function.
    img = cv2.GaussianBlur(img, (5, 5), 0)

    for channel in cv2.split(img):  # GRB

        for threshold in range(0, 255, 26):  # Try with different thresholds
            if threshold == 0:
                # OpenCV Canny edge detection contains already:
                # 1. Gaussian smoothing
                # 2. Derivative (Sobel kernel)
                # 3. Non-maximal suppression
                # 4. Hysteresis thresholding
                binary = cv2.Canny(channel, 0, 50, apertureSize=5)
                binary = cv2.dilate(binary, None)
            else:
                _, binary = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)

            # Find contours
            # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
            # Use CHAIN_APPROX_SIMPLE to remove redundant points of straight edges
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Find rectangles
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and binary.shape[0] * binary.shape[1] * 1e-3 < cv2.contourArea(cnt) < binary.shape[0] * binary.shape[1] * 0.05 \
                        and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([utils.angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        rectangles.append(cnt)

    return rectangles


def rate_rectangles(img: np.ndarray, rectangles: list) -> np.ndarray:
    """
    Score rectangles proportionally to their likelihood of being the scoreboard.

    Scoreboard is usually on the corners, so the further away from the centre of the image, the better.
    Histogram of the scoreboard tends to be flat.
    Alphanumeric characters are within it.

    :param img: input image.
    :param rectangles: list of N rectangles, where each rectangle is a 4 x 2 numpy array containing the vertices with notation [x, y].
    :return: scores for rectangles, of the same length of input list. Each score is a value in range [0,1],
        but the output vector does not represent a pdf yet, for visualization purposes.
    """
    rates = np.zeros(len(rectangles))  # output initialization

    # Centre distance
    img_centre = np.flip(np.array(img.shape[:2]) // 2)  # frame central coordinates [x, y]
    max_dist = np.linalg.norm(img_centre)  # maximum distance from centre (computed between top-left corner [0,0] and img_centre)
    for i, rect in enumerate(rectangles):
        rect_centre = np.mean(rect, axis=0)
        # Get rates proportional to the squared distance to favour edge rectangles
        rates[i] = (np.linalg.norm(rect_centre - img_centre) / max_dist) ** 2
        # ASSUMPTION keep only if in bottom left quadrant
        if rect_centre[0] > img_centre[0] or rect_centre[1] < img_centre[1]:
            rates[i] = 0

    return rates  # TODO return pdf: / np.sum(rates)


if __name__ == "__main__":  # Single image processing

    import argparse
    from random import choice
    import sys
    import utils

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", dest="debug", default=False)
    args = parser.parse_args()

    # Extract frame with available annotations
    annotations = utils.read_annotations(
        os.path.join("..", "data", "top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json"))
    k = choice(list(annotations))
    annotations = annotations[k]                                # bbox: [x0 y0 x1 y1]
    frame = utils.extract_frame(os.path.join("..", "data", "top-100-shots-rallies-2018-atp-season.mp4"), int(k))
    box = utils.extract_box_from_frame(frame, annotations['bbox'])
    if args.debug:
        cv2.imwrite(str(OUTPUT_PATH/"frame.png"), frame)        # Original frame
        cv2.imwrite(str(OUTPUT_PATH/"scoreboard.png"), box)     # Target scoreboard

    # Groundtruth OCR
    print(f"From whole image:\n{pytesseract.image_to_string(frame)}", end="\n\n")
    print(f"From annotated box:\n{pytesseract.image_to_string(box)}", end="\n\n")

    # Extract rectangles
    rectangles = find_rectangles(frame)
    if args.debug:
        frame_out = frame.copy()
        cv2.drawContours(frame_out, rectangles, -1, (0, 255, 0), 3)
        cv2.imwrite(str(OUTPUT_PATH/"frame-rects.png"), frame_out)
        del frame_out

    # Rate rectangles
    # Used to process the rectangles in a meaningful order.
    scores = rate_rectangles(frame, rectangles)
    scores_sorted_index = np.flip(np.argsort(scores))
    if args.debug:
        frame_out = frame.copy()
        for i, rect in enumerate(rectangles):
            cv2.drawContours(frame_out, [rect], -1, [int(scores[i] * 255)] * 3, 5)
        cv2.imwrite(str(OUTPUT_PATH/"frame-rects-scored.png"), frame_out)
        del frame_out

    # Tentative scoreboard identification
    # Tentatives proceed in order of rectangle rating. As soon as a rectangle with text on two lines is found, the loop breaks.
    found = False
    for index_outer, pointer_outer in enumerate(scores_sorted_index):
        print(f"Tentative {index_outer}: {rectangles[pointer_outer]}")

        # Condition 1: there are 2 rows of text to be read from.
        coords = utils.convert_to_rect(rectangles[pointer_outer])
        coords[3] = coords[3] - (coords[3] - coords[1]) // 2
        upper_patch = utils.extract_box_from_frame(frame, coords)
        line1 = pytesseract.image_to_string(upper_patch).strip()
        coords = utils.convert_to_rect(rectangles[pointer_outer])
        coords[1] = coords[1] + (coords[3] - coords[1]) // 2
        bottom_patch = utils.extract_box_from_frame(frame, coords)
        line2 = pytesseract.image_to_string(bottom_patch).strip()
        if args.debug:
            cv2.imwrite(str(OUTPUT_PATH/"scoreboard-upper.png"), upper_patch)
            cv2.imwrite(str(OUTPUT_PATH/"scoreboard-bottom.png"), bottom_patch)
        if len(line1) == 0 or len(line2) == 0:  # no text found on at least one row
            continue

        # ASSUMPTION: the scoreboard contains another, smaller, detected rectangle.
        for index_inner, pointer_inner in enumerate(scores_sorted_index):
            # Skip self
            if index_inner == index_outer:
                continue
            if utils.is_inside(outer_rect=utils.convert_to_rect(rectangles[pointer_outer]),
                               inner_rect=utils.convert_to_rect(rectangles[pointer_inner])):
                print("Found")
                if args.debug:
                    frame_out = frame.copy()
                    cv2.drawContours(frame_out, [rectangles[pointer_outer], rectangles[pointer_inner]], -1, [0, 255, 0], 3)
                    cv2.imwrite(str(OUTPUT_PATH/"frame-rects-contained.png"), frame_out)
                found = True
                break

        if found:
            break

    if not found:
        sys.exit(1)

    print(f"From extracted box:\n{pytesseract.image_to_string(utils.extract_box_from_frame(frame, utils.convert_to_rect(rectangles[pointer_outer])))}")
