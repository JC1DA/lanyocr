import json
import os
from abc import ABC, abstractmethod
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import array
from pydantic import BaseModel
from shapely.geometry import Polygon

from lanyocr import LanyOcr


class DatasetType(str, Enum):
    ICDAR2015 = "ICDAR2015"
    ICDAR2017 = "ICDAR2017"


class DetectorAccuracy(BaseModel):
    IOU: float
    # add more if needed


class RecognizerAccuracy(BaseModel):
    precision: float
    recall: float
    # add more or remove if needed


class OcrAccuracy(BaseModel):
    precision: float
    recall: float
    IOU: float
    # add more or remove if needed


class LanyBenchmarker:
    def __init__(self, ocr: LanyOcr, dataset_path: str) -> None:
        self.dataset = []
        self.metadata = []
        self.ocr = ocr
        self.dataset_path = dataset_path
        self.bounding = []
        self.predict = []

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
        dectector_results = DetectorAccuracy
        IOU_list = []
        for index in self.dataset:
            print(f'{self.dataset_path}/{index["image_name"]}')
            picture = cv2.imread(f'{self.dataset_path}/{index["image_name"]}')
            results = self.ocr.detector.infer(picture)
            polygon_list = []
            for result in results:
                point = result.points
                polygon_list.append(
                    Polygon(
                        [
                            tuple(point[0]),
                            tuple(point[1]),
                            tuple(point[2]),
                            tuple(point[3]),
                        ]
                    )
                )
            for validation_text in index["text"]:
                polygon = validation_text["vertices"]
                iou = 0.0
                IOU_temp = [iou]
                for polygon_2 in polygon_list:
                    if polygon.intersects(polygon_2):
                        intersect = polygon.intersection(polygon_2).area
                        union = polygon.union(polygon_2).area
                        iou = intersect / union
                        IOU_temp.append(iou)
                IOU_list.append(max(IOU_temp))
            self.bounding.append(polygon_list)
        final_IOU = array(IOU_list)
        dectector_results.IOU = final_IOU.mean()
        print(dectector_results.IOU)
        return dectector_results

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
        precision_list = []
        recall_list = []
        recognizer_accuracy = RecognizerAccuracy
        for index in self.dataset:
            text_list = index["text"]
            polygon_list = [
                [text["vertices"], text["transcription"]] for text in text_list
            ]
            picture = cv2.imread(f'{self.dataset_path}/{index["image_name"]}')
            for polygon in polygon_list:
                x, y = polygon[0].exterior.coords.xy
                points = [list(point) for point in zip(x, y)]
                del points[-1]
                crop = np.array([points]).astype(int)
                rect = cv2.boundingRect(crop)
                x, y, w, h = rect
                cropped = picture[y : y + h, x : x + w].copy()
                predict = self.ocr.recognizer.infer(cropped)[0]
                if polygon[1] != "###" and len(predict) != 0:
                    precision = len(
                        [(i, j) for i, j in zip(polygon[1], predict) if i == j]
                    ) / len(predict)
                    recall = len([i for i in predict if i in polygon[1]]) / len(predict)
                    precision_list.append(precision)
                    recall_list.append(recall)
                elif polygon[1] != "###" and len(predict) == 0:
                    precision_list.append(0)
                    recall_list.append(0)
        final_precision = array(precision_list)
        final_recall = array(recall_list)
        recognizer_accuracy.precision = final_precision.mean()
        recognizer_accuracy.recall = final_recall.mean()
        print(recognizer_accuracy.recall, recognizer_accuracy.precision)

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

        accuracy_results = OcrAccuracy
        precision_list = []
        recall_list = []
        IOU_list = []
        for index in self.dataset:
            polygon_list_predict = []
            picture = cv2.imread(f'{self.dataset_path}/{index["image_name"]}')
            text_predict = self.ocr.infer(picture)
            text_list = index["text"]
            polygon_list = [
                [text["vertices"], text["transcription"]] for text in text_list
            ]
            for text in text_predict:
                point = text.line.sub_rrects[0].points
                polygon = Polygon(
                    [tuple(point[0]), tuple(point[1]), tuple(point[2]), tuple(point[3])]
                )
                polygon_list_predict.append([polygon, text.text])
            for polygon in polygon_list:
                precision = 0
                recall = 0
                iou = 0
                if polygon[1] == "###":
                    pass
                else:
                    IOU_temp = [iou]
                    Polygon_temp = []
                    for polygon_2 in polygon_list_predict:
                        if polygon[0].intersects(polygon_2[0]):
                            intersect = polygon[0].intersection(polygon_2[0]).area
                            union = polygon[0].union(polygon_2[0]).area
                            iou = intersect / union
                            IOU_temp.append(iou)
                            Polygon_temp.append(polygon_2)
                    max_IOU = max(IOU_temp)
                    IOU_list.append(max_IOU)
                    max_index = IOU_temp.index(max_IOU)
                    if max_IOU == 0:
                        pass
                    else:
                        polygon_2 = polygon_list_predict[max_index - 1]
                        if len(polygon[1]) != 0:
                            precision = len(
                                [
                                    (i, j)
                                    for i, j in zip(polygon[1], polygon_2[1])
                                    if i == j
                                ]
                            ) / len(polygon_2[1])
                            recall = len(
                                [i for i in polygon_2[1] if i in polygon[1]]
                            ) / len(polygon_2[1])
                        else:
                            precision = 0
                            recall = 0
                    recall_list.append(recall)
                    precision_list.append(precision)
        final_precision = array(precision_list)
        final_recall = array(recall_list)
        final_IOU = array(IOU_list)
        accuracy_results.precision = final_precision.mean()
        accuracy_results.recall = final_recall.mean()
        accuracy_results.IOU = final_IOU.mean()
        print(accuracy_results.recall, accuracy_results.precision, accuracy_results.IOU)
        return accuracy_results



class LanyBenchmarkerICDAR2015(LanyBenchmarker):
    def load_dataset(self):
        f_train = open(f"{self.dataset_path}/metadata/tie_train_v1.json")
        f_val = open(f"{self.dataset_path}/metadata/tie_val_v1.json")
        train_metadata = json.load(f_train)
        val_metadata = json.load(f_val)
        f_train.close()
        f_val.close()
        ICDAR2015_train_metadata = [
            index for index in train_metadata if index["dataset"] == "icdar15"
        ]
        ICDAR2015_val_metadata = [
            index for index in val_metadata if index["dataset"] == "icdar15"
        ]
        ICDAR2015_val_metadata.extend(ICDAR2015_train_metadata)
        for index in ICDAR2015_val_metadata:
            text = index["text"]
            for sample in text:
                polygon_point = [
                    item for points in sample["vertices"] for item in points
                ]
                polygon = Polygon(
                    [
                        (polygon_point[0], polygon_point[1]),
                        (polygon_point[2], polygon_point[3]),
                        (polygon_point[4], polygon_point[5]),
                        (polygon_point[6], polygon_point[7]),
                    ]
                )
                sample["vertices"] = polygon
            index["text"] = text
        self.dataset = ICDAR2015_val_metadata
        print("LOAD DONE")
        return super().load_dataset()


class LanyBenchmarkerICDAR2017(LanyBenchmarker):
    def load_dataset(self):
        f_train = open(f"{self.dataset_path}/metadata/tie_train_v1.json")
        f_val = open(f"{self.dataset_path}/metadata/tie_val_v1.json")
        train_metadata = json.load(f_train)
        val_metadata = json.load(f_val)
        f_train.close()
        f_val.close()
        print(len(train_metadata))
        print(len(val_metadata))
        ICDAR2017_train_metadata = [
            index for index in train_metadata if index["dataset"] == "cocotext"
        ]
        ICDAR2017_val_metadata = [
            index for index in val_metadata if index["dataset"] == "cocotext"
        ]
        ICDAR2017_val_metadata.extend(ICDAR2017_train_metadata)
        del f_train
        del f_val
        for index in ICDAR2017_val_metadata:
            text = index["text"]
            for sample in text:
                polygon_point = [
                    item for points in sample["vertices"] for item in points
                ]
                polygon = Polygon(
                    [
                        (polygon_point[0], polygon_point[1]),
                        (polygon_point[2], polygon_point[3]),
                        (polygon_point[4], polygon_point[5]),
                        (polygon_point[6], polygon_point[7]),
                    ]
                )
                sample["vertices"] = polygon
            index["text"] = text
        self.dataset = ICDAR2017_val_metadata
        print("LOAD DONE")
        return super().load_dataset()


ocr = LanyOcr()
test = LanyBenchmarkerICDAR2015(ocr, "./datasets")
data = test.load_dataset()
print(1)
test.compute_e2e_accuracy()
# METADATA CAN SUPPORT FOLLOWING DATASET
# ['hiertext', 'textocr', 'icdar13', 'icdar15', 'mlt19', 'cocotext', 'openimages']
