import os
from abc import ABC, abstractmethod
from typing import List, Tuple

from lanyocr.lanyocr_utils import LanyOcrRRect

MODULE_DICT = {"easyocr_craft": ["easyocr_craft.py", "EasyOcrCraft"]}


class LanyOcrDetector(ABC):
    def __init__(self, use_gpu: bool = True) -> None:
        self.use_gpu = use_gpu

    @abstractmethod
    def infer(self, image) -> List[LanyOcrRRect]:
        raise NotImplementedError


class LanyOcrDetectorFactory:
    @staticmethod
    def create(name: str, **kwargs) -> LanyOcrDetector:
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
