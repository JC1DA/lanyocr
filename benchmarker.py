from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from lanyocr import LanyOcr

import os

from shapely.geometry import Polygon

import matplotlib.pyplot as plt

from numpy import array
import numpy as np

import seaborn as sns

import cv2


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


class LanyBenchmarker(ABC):
    def __init__(self, ocr: LanyOcr, dataset_path: str) -> None:
        self.dataset = []
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
        name = []
        for filename in os.listdir(self.dataset_path + '/images'):
            if filename.endswith(".jpg"):
                name.append(os.path.join(filename))
            else:
                continue
        name = [x.replace('.jpg', '') for x in name]
        for index in name:
            picture = cv2.imread(f'{self.dataset_path}/images/{index}.jpg')
            with open(f'{self.dataset_path}/validation/gt_{index}.txt', "r") as f:
                data = f.read().replace('ï»¿', '')
                data = data.split('\n')
                data.remove('')
                data.sort()
                polygon_list = []
                for index in data:
                    temp_point_split = index.split(',')
                    temp_point = [float(temp_point_split[x]) for x in range(8)]
                    polygon = Polygon([(temp_point[0], temp_point[1]), (temp_point[2], temp_point[3]),
                                       (temp_point[4], temp_point[5]), (temp_point[6], temp_point[7])])
                    if len(temp_point_split) == 9:
                        polygon_list.append([polygon, temp_point_split[8]])
                    else:
                        polygon_list.append([polygon, ' '])
            self.dataset.append([picture, polygon_list])
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
            results = self.ocr.detector.infer(index[0])
            polygon_list = []
            for result in results:
                point = result.points
                polygon_list.append(Polygon([tuple(point[0]), tuple(point[1]),
                                             tuple(point[2]), tuple(point[3])]))
            for polygon in index[1]:
                flag_2 = False
                if polygon[1] == '###':
                    pass
                else:
                    for polygon_2 in polygon_list:
                        if polygon[0].intersects(polygon_2):
                            flag_2 = True
                            intersect = polygon[0].intersection(polygon_2).area
                            union = polygon[0].union(polygon_2).area
                            iou = intersect / union
                if not flag_2:
                    iou = 0.0
                IOU_list.append(iou)
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
###### NOTE : run self.ocr.recognizer.infer(...) on images can not predict all text
            - return accuracy in the form of RecognizerAccuracy
        """
        # data  = self.dataset[0]
        # temp =self.ocr.recognizer.infer(data[0])
        # pass

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
            text_predict = self.ocr.infer(index[0])
            for text in text_predict:
                point = text.line.sub_rrects[0].points
                polygon = Polygon([tuple(point[0]), tuple(point[1]),
                                   tuple(point[2]), tuple(point[3])])
                polygon_list_predict.append([polygon, text.text])
            for polygon in index[1]:
                flag_2 = False
                if polygon[1] == '###':
                    pass
                else:
                    for polygon_2 in polygon_list_predict:
                        if polygon[0].intersects(polygon_2[0]):
                            flag_2 = True
                            intersect = polygon[0].intersection(polygon_2[0]).area
                            union = polygon[0].union(polygon_2[0]).area
                            iou = intersect / union
                            if len(polygon[1]) != 0:
                                precision = len([(i, j) for i, j in zip(polygon[1], polygon_2[1]) if i == j]) / len(
                                    polygon_2[1])
                                recall = len([i for i in polygon_2[1] if i in polygon[1]]) / len(polygon_2[1])
                            else:
                                precision = 0
                                recall = 0
                    if not flag_2:
                        precision = 0
                        recall = 0
                        iou = 0
                    precision_list.append(precision)
                    recall_list.append(recall)
                    IOU_list.append(iou)
        final_precision = array(precision_list)
        final_recall = array(recall_list)
        final_IOU = array(IOU_list)
        accuracy_results.precision = final_precision.mean()
        accuracy_results.recall = final_recall.mean()
        accuracy_results.IOU = final_IOU.mean()
        print(accuracy_results.recall, accuracy_results.precision,accuracy_results.IOU)
        return accuracy_results


class LanyBenchmarkerICDAR2015(LanyBenchmarker):
    def load_dataset(self):
        # TODO: implement load function here
        return super().load_dataset()


class LanyBenchmarkerICDAR2017(LanyBenchmarker):
    def load_dataset(self):
        # TODO: implement load function here
        return super().load_dataset()


ocr = LanyOcr()
test = LanyBenchmarkerICDAR2015(ocr, './ICDAR/2015')
data = test.load_dataset()
test.compute_recognizer_accuracy()
print(1)
