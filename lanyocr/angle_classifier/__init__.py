import os
from abc import ABC, abstractmethod
from typing import List, Tuple

MODULE_DICT = {
    "paddleocr_mobile": [
        "paddleocr_angle_classifier_v2.py",
        "PaddleOcrAngleClassifierv2",
    ]
}


class LanyOcrAngleClassifier(ABC):
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu

    @abstractmethod
    def infer(self, bgr_image, thresh=0.5) -> int:
        raise NotImplementedError

    @abstractmethod
    def infer_batch(self, bgr_images, thresh=0.5) -> List[int]:
        raise NotImplementedError


class LanyOcrAngleClassifierFactory:
    @staticmethod
    def create(name: str, **kwargs) -> LanyOcrAngleClassifier:
        if name not in MODULE_DICT:
            raise ValueError("Invalid name")

        import importlib, inspect

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        module_name = name
        module_path, class_name = MODULE_DICT[name]
        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(cur_dir, module_path)
        )
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)

        clsmembers = inspect.getmembers(_module, inspect.isclass)
        for clsmember in clsmembers:
            name, cls_def = clsmember
            if name == class_name:
                return cls_def(**kwargs)

        raise ValueError(f"Could not find class {class_name} in {module_path}")
