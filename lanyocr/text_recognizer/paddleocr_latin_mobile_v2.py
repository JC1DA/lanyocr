import os
import json
import numpy as np
from typing import Tuple, List
from lanyocr.text_recognizer.paddleocr_base import PaddleOcrBase
from lanyocr.utils import download_model


class PaddleOcrLatinMobile(PaddleOcrBase):
    def __init__(self, use_gpu: bool = True) -> None:
        print("Recognizer: PaddleOcrLatinMobile")

        model_h = 32
        model_w = 320

        cur_dir = os.path.dirname(os.path.realpath(__file__))

        model_ignored_tokens = [0]
        accepted_characters = []
        model_characters = json.load(
            open(os.path.join(cur_dir, "dicts/paddleocr_latin_dict.json"))
        )
        model_path = download_model("lanyocr_paddleocr_latin_mobile_v2.onnx")

        assert os.path.exists(model_path)

        super().__init__(
            use_gpu,
            model_ignored_tokens,
            model_characters,
            accepted_characters,
            model_path,
            model_w,
            model_h,
        )
