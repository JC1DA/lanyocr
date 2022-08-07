from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from lanyocr import LanyOcr
from pydantic import BaseModel


class DatasetType(str, Enum):
    ICDAR2015 = "ICDAR2015"
    ICDAR2017 = "ICDAR2017"


class DetectorAccuracy(BaseModel):
    precision: float
    recall: float
    hmean: float


class RecognizerAccuracy(BaseModel):
    precision: float
    recall: float
    hmean: float


class OcrAccuracy(BaseModel):
    precision: float
    recall: float
    hmean: float


class LanyBenchmarker(ABC):
    def __init__(self, ocr: LanyOcr, dataset_path: str) -> None:
        self.dataset = []
        self.ocr = ocr
        self.dataset_path = dataset_path
        self.bounding = []
        self.predict = []
        self.chunk_size = 0

        # use ICDAR 2015 format for all kind of dataset
        self.id2imagePath: Dict[int, str] = {}
        self.id2gts: Dict[int, bytes] = {}
        self.id2preds: Dict[int, bytes] = {}

        self.load_dataset()

    @abstractmethod
    def load_dataset(self):
        """Load the dataset from the dataset_path into ICDAR 2015 format.
        Raises:
            NotImplementedError: _description_
        """

        raise NotImplementedError

    def compute_detector_accuracy(self) -> DetectorAccuracy:
        """Compute accuracy of only text detector
        Returns:
            DetectorAccuracy: _description_
        """

        """
            TODO: 
            - call self.ocr.detector.infer(...) on entire dataset to get bounding boxes of texts
            - compute precision/recall/fscore
            - return accuracy in the form of DetectorAccuracy
        """

        # run detector on every single image and store results as a string
        for id in self.id2imagePath:
            image_path = self.id2imagePath[id]
            image = cv2.imread(image_path)
            det_results = self.ocr.detector.infer(image)

            result_str = ""
            for det_result in det_results:
                points = det_result.points
                if len(points) != 4:
                    pass

                result_str += "{},{},".format(int(points[0][0]), int(points[0][1]))
                result_str += "{},{},".format(int(points[1][0]), int(points[1][1]))
                result_str += "{},{},".format(int(points[2][0]), int(points[2][1]))
                result_str += "{},{}".format(int(points[3][0]), int(points[3][1]))
                result_str += "\n"

            self.id2preds[id] = result_str.encode()

        from .detector_utils import evaluate_method

        results = evaluate_method(self.id2gts, self.id2preds)

        return DetectorAccuracy(
            precision=results["method"]["precision"],
            recall=results["method"]["recall"],
            hmean=results["method"]["hmean"],
        )

    def compute_recognizer_accuracy(self) -> RecognizerAccuracy:
        """Compute accuracy of only text recognizer
        Returns:
            RecognizerAccuracy: _description_
        """

        """ 
            TODO:
            - Assume the detector extracts all the box locations from dataset, extract all text images from dataset labels
            - run self.ocr.recognizer.infer(...) on all text images to predict text
            - return accuracy in the form of RecognizerAccuracy
        """

        _id2preds = {}

        for id in self.id2imagePath:
            image_path = self.id2imagePath[id]
            image = cv2.imread(image_path)

            result_str = ""
            gts = self.id2gts[id].decode().split("\n")
            for gt_idx, gt in enumerate(gts):
                if gt == "":
                    continue

                parts = gt.split(",")
                if parts[8] == "###":
                    # don't care
                    continue

                points = np.array([int(s) for s in parts[:8]]).reshape([1, 4, 2])
                rect = cv2.minAreaRect(points)
                # sub_img = crop_rect(image, rect)
                sub_img = self._crop_rect(image, rect)

                h, w = sub_img.shape[:2]

                if h == 0 or w == 0:
                    continue

                if h > 1.5 * w:
                    sub_img = cv2.rotate(sub_img, cv2.ROTATE_90_CLOCKWISE)

                # if self.ocr.angle_classifier.infer(sub_img) == 180:
                #     sub_img = cv2.rotate(sub_img, cv2.ROTATE_180)

                text_pred, _ = self.ocr.recognizer.infer(sub_img)

                # print(id, gt, text_pred)
                # if parts[8] != text_pred:
                #     print(id, gt, text_pred)
                #     # cv2.imwrite(f"./tmp/{id}_{gt_idx}.jpg", sub_img)

                for i in range(8):
                    result_str += parts[i] + ","
                result_str += text_pred + "\n"

            _id2preds[id] = result_str.encode()

        from .ocr_utils import evaluate_method

        results = evaluate_method(self.id2gts, _id2preds)

        return RecognizerAccuracy(
            precision=results["method"]["precision"],
            recall=results["method"]["recall"],
            hmean=results["method"]["hmean"],
        )

    def compute_e2e_accuracy(self) -> OcrAccuracy:
        """Compute the end-to-end accuracy of the ocr
        Returns:
            OcrAccuracy: _description_
        """

        """
            TODO:
            - run self.ocr.infer(...) on the entire dataset
            - return the accuracy in form of OcrAccuracy
        """

        # disable merge inference when running benchmark
        self.ocr.merge_boxes_inference = False

        for id in self.id2imagePath:
            image_path = self.id2imagePath[id]
            image = cv2.imread(image_path)

            ocr_lines = self.ocr.infer(image)

            result_str = ""
            for line in ocr_lines:
                for det_result in line.line.sub_rrects:
                    points = det_result.points
                    if len(points) != 4:
                        pass

                    result_str += "{},{},".format(int(points[0][0]), int(points[0][1]))
                    result_str += "{},{},".format(int(points[1][0]), int(points[1][1]))
                    result_str += "{},{},".format(int(points[2][0]), int(points[2][1]))
                    result_str += "{},{},".format(int(points[3][0]), int(points[3][1]))
                    result_str += det_result.text + "\n"

            self.id2preds[id] = result_str.encode()

        from .ocr_utils import evaluate_method

        results = evaluate_method(self.id2gts, self.id2preds)

        return OcrAccuracy(
            precision=results["method"]["precision"],
            recall=results["method"]["recall"],
            hmean=results["method"]["hmean"],
        )

    def _crop_rect(self, img, rect):
        # get the parameter of the small rectangle
        center = rect[0]
        size = rect[1]
        angle = rect[2]
        center, size = tuple(map(int, center)), tuple(map(int, size))

        # get row and col num in img
        rows, cols = img.shape[0], img.shape[1]

        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (cols, rows))
        out = cv2.getRectSubPix(img_rot, size, center)

        return out
