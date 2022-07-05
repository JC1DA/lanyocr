import json
import math
import os
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime
from lanyocr.lanyocr_utils import download_model
from lanyocr.text_recognizer import LanyOcrRecognizer


class MMOCR(LanyOcrRecognizer):
    def __init__(
        self,
        use_gpu: bool = True,
        model_path: str = "",
        model_w: int = 100,
        model_h: int = 32,
    ) -> None:
        super().__init__(use_gpu)

        self.model_w = model_w
        self.model_h = model_h
        self.model_path = model_path

        # load model
        providers = ["CPUExecutionProvider"]
        if self.use_gpu:
            providers = ["CUDAExecutionProvider"]

        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )

        self.session = onnxruntime.InferenceSession(
            self.model_path, sess_options=opts, providers=providers
        )

    def infer(self, bgr_img) -> Tuple[str, float]:
        norm_img = self.normalize_img(bgr_img)

        pred = self.session.run(None, {"input": [norm_img]})[0][0]

        return self.decode(pred)

    def infer_batch(self, bgr_images) -> List[Tuple[str, float]]:
        batch_results = []

        normed_imgs = [self.normalize_img(img) for img in bgr_images]

        for idx in range(0, len(normed_imgs), self.max_batch_size):
            batch_preds = self.session.run(
                None, {"input": normed_imgs[idx : idx + self.max_batch_size]}
            )[0]

            for pred in batch_preds:
                result = self.decode(pred)
                batch_results.append(result)

        return batch_results

    def get_model_height(self) -> int:
        return self.model_h

    def get_model_width(self) -> int:
        return self.model_w

    def normalize_img(self, bgr_img):
        raise NotImplementedError

    def decode(self, pred) -> Tuple[str, float]:
        raise NotImplementedError
