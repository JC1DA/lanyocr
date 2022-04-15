from pydantic import BaseModel
from enum import Enum
from typing import Any, List, Tuple
import cv2
import math
import numpy as np


class LanyOcrMergeType(str, Enum):
    UPWARD = "upward"
    DOWNWARD = "downward"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class LanyOcrBoundingBox(BaseModel):
    left: float
    top: float
    right: float
    bottom: float

    def area(self) -> float:
        return (self.right - self.left) * (self.bottom - self.top)


class LanyOcrRRect(BaseModel):
    rrect: Any
    points: List[List[float]]
    direction: str

    def getCenter(self) -> Tuple[float, float]:
        return (self.rrect[0][0], self.rrect[0][1])

    def getCenterX(self) -> float:
        return self.getCenter()[0]

    def getCenterY(self) -> float:
        return self.getCenter()[1]

    def getWidth(self) -> float:
        return self.rrect[1][0]

    def getHeight(self) -> float:
        return self.rrect[1][1]

    def getAngle(self) -> float:
        return self.rrect[2]

    def minX(self) -> float:
        return np.min([p[0] for p in self.points])

    def maxX(self) -> float:
        return np.max([p[0] for p in self.points])

    def minY(self) -> float:
        return np.min([p[1] for p in self.points])

    def maxY(self) -> float:
        return np.max([p[1] for p in self.points])

    def getBoundingBox(self) -> LanyOcrBoundingBox:
        box = cv2.boundingRect(
            np.expand_dims(np.array(self.points).astype(np.int32), 0)
        )
        return LanyOcrBoundingBox(
            left=box[0], top=box[1], right=box[0] + box[2], bottom=box[1] + box[3]
        )


class LanyOcrDirectionalBoxGroups(BaseModel):
    upward: List[LanyOcrRRect]
    downward: List[LanyOcrRRect]
    vertical: List[LanyOcrRRect]
    horizontal: List[LanyOcrRRect]


class LanyOcrTextLine(BaseModel):
    # rrect: LanyOcrRRect
    sub_rrects: List[LanyOcrRRect]

    def avgAngle(self) -> float:
        angles = [rrect.rrect[2] for rrect in self.sub_rrects]
        return np.mean(angles)

    def avgWidth(self) -> float:
        widths = [rrect.rrect[1][0] for rrect in self.sub_rrects]
        return np.mean(widths)

    def avgHeight(self) -> float:
        heights = [rrect.rrect[1][1] for rrect in self.sub_rrects]
        return np.mean(heights)

    def get_rrect(self):
        points = []
        for rrect in self.sub_rrects:
            points.extend(rrect.points)
        return cv2.minAreaRect(np.array([points]).astype(np.int32))


def iou(box1: LanyOcrBoundingBox, box2: LanyOcrBoundingBox) -> float:
    xx1 = max(box1.left, box2.left)
    yy1 = max(box1.top, box2.top)
    xx2 = min(box1.right, box2.right)
    yy2 = min(box1.bottom, box2.bottom)

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h

    box1_area = box1.area()
    box2_area = box2.area()

    return inter / (box1_area + box2_area - inter)


def compute_angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product) / np.pi * 180


def check_area_merge(rrect1: LanyOcrRRect, rrect2: LanyOcrRRect, thresh=0.75):
    points = []
    points.extend(rrect1.points)
    points.extend(rrect2.points)

    merged_rrect = cv2.minAreaRect(np.array([points]).astype(np.int32))

    rrect1_area = rrect1.getHeight() * rrect1.getWidth()
    rrect2_area = rrect2.getHeight() * rrect2.getWidth()
    merged_rrect_area = merged_rrect[1][0] * merged_rrect[1][1]

    ratio = (rrect1_area + rrect2_area) / merged_rrect_area
    return ratio > thresh


def check_merged_size(text_line: LanyOcrTextLine, rrect: LanyOcrRRect):
    text_line_rrect = text_line.get_rrect()
    text_line_points = cv2.boxPoints(text_line_rrect)

    points = []
    points.extend(text_line_points)
    points.extend(rrect.points)

    merged_rrect = cv2.minAreaRect(np.array([points]).astype(np.int32))
    w, h = merged_rrect[1]

    # merged rect should not have both h & w larger than each rrect
    if w > 1.5 * text_line_rrect[1][0] and h > 1.5 * text_line_rrect[1][1]:
        return False

    if w > 1.5 * rrect.getWidth() and h > 1.5 * rrect.getHeight():
        return False

    return True


def crop_rect(img, rrect):
    points = cv2.boxPoints(rrect)
    box = np.int0(np.array(points))

    min_x = np.min(box[:, 0])
    max_x = np.max(box[:, 0])
    min_y = np.min(box[:, 1])
    max_y = np.max(box[:, 1])

    cropped_img = img[min_y:max_y, min_x:max_x, :]
    return np.array(cropped_img)


# def crop_rrect(img, rrect, angle_thresh=5):
#     if rrect[2] in [0, 90]:
#         return crop_rect(img, rrect)

#     # if rrect[2] < angle_thresh or rrect[2] > 90 - angle_thresh:
#     #     return crop_rect(img, rrect)

#     padding_size = int(max(rrect[1][0], rrect[1][1])) // 2
#     padded_img = cv2.copyMakeBorder(
#         img,
#         padding_size,
#         padding_size,
#         padding_size,
#         padding_size,
#         cv2.BORDER_CONSTANT,
#     )

#     m = cv2.getRotationMatrix2D(
#         (rrect[0][0] + padding_size, rrect[0][1] + padding_size), rrect[2], 1
#     )

#     rotated_img = cv2.warpAffine(
#         padded_img, m, (padded_img.shape[1], padded_img.shape[0])
#     )

#     rect = cv2.getRectSubPix(
#         rotated_img,
#         (int(rrect[1][0]), int(rrect[1][1])),
#         (int(rrect[0][0] + padding_size), int(rrect[0][1] + padding_size)),
#     )

#     return rect


def crop_rrect(img, rrect):
    if rrect[2] in [0, 90]:
        return crop_rect(img, rrect)

    padding_size = int(max(rrect[1][0], rrect[1][1])) // 2
    padded_img = cv2.copyMakeBorder(
        img,
        padding_size,
        padding_size,
        padding_size,
        padding_size,
        cv2.BORDER_CONSTANT,
    )

    m = cv2.getRotationMatrix2D(
        (rrect[0][0] + padding_size, rrect[0][1] + padding_size), rrect[2], 1
    )

    rotated_img = cv2.warpAffine(
        padded_img, m, (padded_img.shape[1], padded_img.shape[0])
    )

    rect = cv2.getRectSubPix(
        rotated_img,
        (int(rrect[1][0]), int(rrect[1][1])),
        (int(rrect[0][0] + padding_size), int(rrect[0][1] + padding_size)),
    )

    return rect


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def isascii(s):
    """Check if the characters in string s are in ASCII, U+0-U+7F."""
    return len(s.encode()) == len(s)


def resize_img_to_height(img, model_height=32):
    h = img.shape[0]
    w = img.shape[1]

    ratio = model_height / float(h)
    _w = int(ratio * w)
    return cv2.resize(img, (_w, model_height))


def resize_img_to_width(img, model_width=320):
    h = img.shape[0]
    w = img.shape[1]

    ratio = model_width / float(w)
    _h = int(ratio * h)
    return cv2.resize(img, (model_width, h))


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
