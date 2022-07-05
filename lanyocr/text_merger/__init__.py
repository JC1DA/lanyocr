import os
from abc import ABC, abstractmethod
from typing import List, Tuple
from lanyocr.lanyocr_utils import LanyOcrRRect, LanyOcrTextLine

MODULE_DICT = {
    "lanyocr_craftbased": ["lanyocr_craft_merger.py", "LanyOcrCraftBasedMerger"]
}


class LanyOcrMerger(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def merge_to_lines(
        self,
        rrects: List[LanyOcrRRect],
        merge_rotated: bool = True,
        merge_vertical: bool = True,
    ) -> List[LanyOcrTextLine]:
        raise NotImplementedError


class LanyOcrMergerFactory:
    @staticmethod
    def create(name: str, **kwargs) -> LanyOcrMerger:
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
