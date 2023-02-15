import json
import os
from typing import List
from typing import Tuple

import cv2
import numpy as np

from lanyocr.lanyocr_utils import download_model
from lanyocr.text_recognizer.paddleocr_base import PaddleOcrBase


class PaddleOcrEnPPOCRV3(PaddleOcrBase):
    def __init__(self, use_gpu: bool = True) -> None:
        print("Recognizer: PaddleOcrChv3")

        model_h = 48
        model_w = 320

        cur_dir = os.path.dirname(os.path.realpath(__file__))

        model_ignored_tokens = [0]
        accepted_characters = json.load(
            open(os.path.join(cur_dir, "dicts/paddleocr_en_dict.json"))
        )
        # accepted_characters = []
        model_characters = json.load(
            open(os.path.join(cur_dir, "dicts/paddleocr_en_dict.json"))
        )
        model_path = download_model("lanyocr-en-ppocrv3.onnx")

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

        self.set_max_batch_size(1)

    def normalize_img(self, bgr_img):
        h, w = bgr_img.shape[:2]

        # scale based on h
        ratio = float(self.model_h) / h
        resized_w = int(w * ratio)

        resized_image = cv2.resize(bgr_img, (resized_w, self.model_h))

        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5

        return resized_image
