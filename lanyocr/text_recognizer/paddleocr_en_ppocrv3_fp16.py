import json
import os
from typing import List
from typing import Tuple

import cv2
import numpy as np

from lanyocr.lanyocr_utils import download_model
from lanyocr.text_recognizer.paddleocr_base import PaddleOcrBase


class PaddleOcrEnPPOCRV3FP16(PaddleOcrBase):
    def __init__(self, use_gpu: bool = True) -> None:
        print("Recognizer: PaddleOcrChv3-FP16")
        assert use_gpu, "FP16 model only supports GPU inference."

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
        model_path = download_model("lanyocr-en-ppocrv3_FP16.onnx")

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

        self.max_w = 1024
        self.set_max_batch_size(1)

    # override the base class method
    def infer(self, bgr_img) -> Tuple[str, float]:
        norm_img = self.normalize_img(bgr_img).astype(np.float16)

        preds = self.session.run(None, {"x": [norm_img]})[0][0].astype(np.float32)

        preds_idx = preds.argmax(axis=1)
        preds_prob = preds.max(axis=1)
        results = self.decode([preds_idx], [preds_prob], True)
        text, prob = results[0]
        return text, prob

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

        if resized_w < self.max_w:
            padded_image = np.zeros([3, self.model_h, self.max_w], dtype=np.float32)
            padded_image[:, :, :resized_w] = resized_image
        else:
            padded_image = resized_image

        return resized_image
