import json
import math
import os

import cv2
import numpy as np

# Defaults
ANNOTATIONS = os.path.join("data", "top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json")
VIDEO = os.path.join("data", "top-100-shots-rallies-2018-atp-season.mp4")
HEIGHT = 1080
WIDTH = 1920
FPS = 25
CODEC_OUT = cv2.VideoWriter_fourcc(*"mp4v")
VIDEO_OUT = os.path.join("data", "top-100-shots-output.mp4")


def read_annotations(filename=ANNOTATIONS) -> dict:
    """
    Read and parse annotations file.

    :param filename: path to json file containing annotations.
    :return: annotations dictionary, with keys (str) indicating the frame number, and fields bbox, serving_player,
        name_1, name_2, score_1, score_2.
    """
    with open(filename, 'r') as input_file:
        annotations = json.loads(input_file.read())
    return annotations


def extract_frame(filename=VIDEO, index=0) -> np.ndarray:
    """
    Extract index-th frame from video.

    :param filename: path to video.
    :param index: frame index. If out of bounds, selected randomly.
    :return: frame with opencv convention (BGR).
    """
    # Prevent function from receiving str keys as index
    if isinstance(index, str):
        index = int(index)
        print("Warning: extract_frame method needs an integer, string passed.")
    cap = cv2.VideoCapture(filename)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not 0 <= index < n_frames:
        index = np.random.randint(0, n_frames - 1)
    cap.set(1, index)
    ok, frame = cap.read()
    cv2.imwrite(os.path.join("data", "sample.jpg"), frame)
    cap.release()
    return frame


def extract_box_from_frame(frame: np.ndarray, bbox: list) -> np.ndarray:
    """
    Extract rectangle from image, provided a bounding box.

    :param frame: input frame.
    :param bbox: 4-element list containing [x0 y0 x1 y1].
    :return: bounded part of the frame.
    """
    return frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]


def convert_to_rect(rectangle: np.ndarray) -> np.ndarray:
    """
    Transform rectangle description from 4 [x, y] points into [x0 y0 x1 y1] (TL, BR).

    :param rectangle: array of shape [4, 2], where rows are of type [x, y].
    :return: 4-element array of type [x0 y0 x1 y1]
    """
    return np.concatenate([np.min(rectangle, axis=0), np.max(rectangle, axis=0)])


def is_inside(outer_rect: np.ndarray, inner_rect: np.ndarray, tolerance=3) -> bool:
    """
    Returns whether inner_rect is inside outer_rect, with a margin.

    :param outer_rect: 4-element list containing [x0 y0 x1 y1].
    :param inner_rect: 4-element list containing [x0 y0 x1 y1].
    :param tolerance: tolerance in pixels.
    :return: whether inner_rect is contained in outer_rect.
    """
    return (inner_rect[:2] >= outer_rect[:2] - tolerance).all() and \
        (inner_rect[2:] <= outer_rect[2:] + tolerance).all()


'''
def write_video(filename=VIDEO, annotations=None, estimates=None):
    """Re-write video with annotations"""
    # Video IO
    cap = cv2.VideoCapture(filename)
    out = cv2.VideoWriter(VIDEO_OUT, CODEC_OUT, FPS, (WIDTH, HEIGHT), isColor=True)
    assert cap.isOpened(), print(f"Could not open {filename}")
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Total number of frames: {total_frames}. Processing {total_frames / 43.247}")
    # Loop through frames
    idx = 0
    while cap.isOpened() and idx < total_frames / 43.247:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            out.release()
        out.write(frame)
        # Draw annotation if available
        if annotations is not None and annotations.get(str(idx)) is not None:
            cv2.drawContours(frame, annotations[str(idx)]["bbox"], -1, (0, 255, 0), 3)
        idx += 1
    cap.release()
    out.release()
'''


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
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


class Scoreboard:

    def __init__(self, coordinates: np.ndarray) -> None:
        """
        Keeps track of the scoreboard frame position and provides methods to extract information from it.

        :param coords: vector containing coordinates of the overlay. Two options are available:
            - the first is a vector of shape [4, 2] where the first axis identifies the 4 corner points and
                the second axis the point coordinates as [x, y]. In this case, the smallest rectangle containing all points is kept;
            - the second is a 4-element vector containing [x0, y0, x1, y1], describing a rectangle with top-left corner
                [x0, y0] and bottom right corner [x1, y1].
        """
        # Check input coordinates shape
        if isinstance(coordinates, list):
            try:
                coordinates = np.array(coordinates)
            except _ as e:
                print(e)
        if coordinates.shape == (4, 2):
            self.position = convert_to_rect(coordinates)
        elif coordinates.shape == (4,):
            # TODO check top-left and bottom-right validity
            self.position = coordinates

        # Convert position to array of integers
        # keeping the fractional parts of each point within the defined rectangle.
        self.position[:2] = [np.floor(coord) for coord in self.position[:2]]
        self.position[2:] = [np.ceil(coord) for coord in self.position[2:]]
        self.position = self.position.astype(dtype=np.int32)
    
    @property
    def top_left(self):
        return self.position[:2]

    @property
    def bottom_right(self):
        return self.position[2:]

    @property
    def width(self):
        return self.position[2] - self.position[0] + 1

    @property
    def height(self):
        return self.position[3] - self.position[1] + 1
    
    def as_rect(self) -> np.ndarray:
        """
        Return array describing the rectangle in xywh format.

        :return: array with [x0, y0, width, height].
        """
        return np.array([*self.position[:2], self.width, self.height], dtype=int)

    def get_overlay_from_frame(self, frame: np.ndarray, player=0) -> np.ndarray:
        """
        Extract scoreboard from provided frame.

        :param frame: image frame from which to extract the scoreboard patch.
        :param player: Score is assumed to contain information divided into 2 vertically-stacked rows, of equal height.
            This parameter can assume three options:
            0: entire overlay;
            1: upper half (player 1);
            2: bottom half (player 2).
        :return: scoreboard frame portion.
        """
        coords = list(self.position)
        if player == 1:  # upper half
            coords[3] = coords[3] - (coords[3] - coords[1]) // 2
        if player == 2:  # bottom half
            coords[1] = coords[1] + (coords[3] - coords[1]) // 2
        return extract_box_from_frame(frame, coords)
