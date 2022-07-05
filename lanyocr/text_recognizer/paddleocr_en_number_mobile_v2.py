import json
import os
from typing import List, Tuple

import numpy as np
from lanyocr.lanyocr_utils import download_model
from lanyocr.text_recognizer.paddleocr_base import PaddleOcrBase


class PaddleOcrEnNumberMobile(PaddleOcrBase):
    def __init__(self, use_gpu: bool = True) -> None:
        print("Recognizer: PaddleOcrEnNumberMobile")

        model_h = 32
        model_w = 320

        cur_dir = os.path.dirname(os.path.realpath(__file__))

        model_ignored_tokens = [0]
        accepted_characters = []
        model_characters = json.load(
            open(os.path.join(cur_dir, "dicts/paddleocr_en_dict.json"))
        )
        # model_path = os.path.join(
        #     cur_dir, "../models/lanyocr-paddleocr-en_number_mobile_v2.0.onnx"
        # )
        model_path = download_model("lanyocr-paddleocr-en_number_mobile_v2.0.onnx")

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

        self.set_max_batch_size(8)
