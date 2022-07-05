import os
from abc import ABC, abstractmethod
from typing import List, Tuple

MODULE_DICT = {
    "paddleocr_en_server": ["paddleocr_ch_ppocr_server_v2.py", "PaddleOcrChServer"],
    "paddleocr_en_mobile": [
        "paddleocr_en_number_mobile_v2.py",
        "PaddleOcrEnNumberMobile",
    ],
    "paddleocr_french_mobile": [
        "paddleocr_french_mobile_v2.py",
        "PaddleOcrFrenchMobile",
    ],
    "paddleocr_latin_mobile": [
        "paddleocr_latin_mobile_v2.py",
        "PaddleOcrLatinMobile",
    ],
    "mmocr_satrn": ["mmocr_satrn.py", "MMOCR_Satrn"],
    "mmocr_satrn_sm": ["mmocr_satrn_sm.py", "MMOCR_Satrn_Sm"],
}


class LanyOcrRecognizer(ABC):
    def __init__(self, use_gpu: bool = True) -> None:
        self.use_gpu: bool = use_gpu
        self.max_batch_size: int = 1

    @abstractmethod
    def infer(self, image) -> Tuple[str, float]:
        raise NotImplementedError

    @abstractmethod
    def infer_batch(self, images) -> List[Tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_model_height(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_model_width(self) -> int:
        raise NotImplementedError

    def set_max_batch_size(self, batch_size: int):
        self.max_batch_size = batch_size


class LanyOcrRecognizerFactory:
    @staticmethod
    def create(name: str, **kwargs) -> LanyOcrRecognizer:
        if name not in MODULE_DICT:
            raise ValueError("Invalid name")

        import importlib
        import inspect

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
