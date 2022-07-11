from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from lanyocr import LanyOcr


class DatasetType(str, Enum):
    ICDAR2015 = "ICDAR2015"
    ICDAR2017 = "ICDAR2017"


class DetectorAccuracy(BaseModel):
    precision: float
    recall: float
    fscore: float
    # add more if needed


class RecognizerAccuracy(BaseModel):
    precision: float
    recall: float
    fscore: float
    # add more or remove if needed


class OcrAccuracy(BaseModel):
    precision: float
    recall: float
    fscore: float
    # add more or remove if needed


class LanyBenchmarker(ABC):
    def __init__(self, ocr: LanyOcr, dataset_path: str) -> None:
        self.ocr = ocr
        self.dataset_path = dataset_path

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

        return None

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

        return None

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

        return None


class LanyBenchmarkerICDAR2015(LanyBenchmarker):
    def load_dataset(self):
        # TODO: implement load function here
        return super().load_dataset()


class LanyBenchmarkerICDAR2017(LanyBenchmarker):
    def load_dataset(self):
        # TODO: implement load function here
        return super().load_dataset()
