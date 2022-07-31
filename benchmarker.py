import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from glob import glob
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import array
from pydantic import BaseModel
from shapely.geometry import Polygon

from lanyocr import LanyOcr
from lanyocr.lanyocr_utils import crop_rect


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
    # add more or remove if needed


class OcrAccuracy(BaseModel):
    precision: float
    recall: float
    IOU: float
    # add more or remove if needed


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

    @abstractmethod
    def load_dataset(self):
        """Load the dataset from the dataset_path based on the format of the dataset.
        Raises:
            NotImplementedError: _description_
        """

        """
            TODO: 
            implement the subclass of this class for each of dataset type (check LanyBenchmarkerICDAR2015 below) 
                to load the dataset into memory correctly.             
            For example, load the entire dataset into a dictionary to map from file path to label data
                self.dataset: Dict[str, List[float]] = {}
            Choose whatever data structure if it fits.
            All the datasets should share the same functions to compute accuracy.
        """

        # FIXME: Move this to LanyBenchmarkerICDAR2015, this method should not be implemented.
        # what if we need to do benchmark on another dataset with different format???
        # Nothing to do with /images folder , about /validation folder we should restrict user to follow some ICDAR format (json file).

        # FIXME: think about a good data-structure so we can reuse the output of this for many different types of dataset
        # Data-structure : I'm thinking about ICDAR2017 data stuctured, their use a json file for text, bounding boxes and more, ...
        # ICDAR 2022 also use json file
        # Now i will keep the old data-structure while considering ICDAR 2017/2022 data-structured

        # FIXME: we are loading all images into memory? what if the dataset is too big? SOLUTION : CHUNKING, IMPLEMENTING
        #  Suggestion : Predict a large dataset will take a lots of time, should we use multithread when user's
        #  computer lack of GPU?

        # raise NotImplementedError

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
            det_results = ocr.detector.infer(image)

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

        from decoder_utils import evaluate_method

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

        from ocr_utils import evaluate_method

        results = evaluate_method(self.id2gts, _id2preds)

        return DetectorAccuracy(
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
        ocr.merge_boxes_inference = False

        for id in self.id2imagePath:
            image_path = self.id2imagePath[id]
            image = cv2.imread(image_path)

            ocr_lines = ocr.infer(image)

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

        from ocr_utils import evaluate_method

        results = evaluate_method(self.id2gts, self.id2preds)

        return DetectorAccuracy(
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


class LanyBenchmarkerICDAR2015(LanyBenchmarker):
    def load_dataset(self):
        image_paths_dir = os.path.join(self.dataset_path, "images")
        gt_paths_dir = os.path.join(self.dataset_path, "validation")

        image_paths = glob(image_paths_dir + "/*")
        gt_paths = glob(gt_paths_dir + "/*")

        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            image_id = int(image_name.split(".")[0].split("_")[-1])
            self.id2imagePath[image_id] = image_path

        for gt_path in gt_paths:
            gt_name = os.path.basename(gt_path)
            gt_id = int(gt_name.split(".")[0].split("_")[-1])
            self.id2gts[gt_id] = (
                open(gt_path, "r", encoding="utf-8-sig").read().encode()
            )

            # for testing
            # lines = open(gt_path, "r", encoding="utf-8-sig").readlines()
            # preds = ""
            # for line in lines:
            #     _parts = line.split(",")
            #     _line = ""
            #     for i in range(9):
            #         _line += _parts[i]
            #         if i < 8:
            #             _line += ","
            #     # preds += _line + "\n"
            #     preds += _line
            # self.id2preds[gt_id] = preds.encode()


ocr = LanyOcr()
test = LanyBenchmarkerICDAR2015(ocr, "./datasets/ICDAR/2015")
data = test.load_dataset()
det_accuracy = test.compute_recognizer_accuracy()
print(det_accuracy)
