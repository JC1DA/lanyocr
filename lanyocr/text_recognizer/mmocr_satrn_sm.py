import json
import math
import os
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime
from lanyocr.lanyocr_utils import download_model
from lanyocr.text_recognizer import LanyOcrRecognizer
from lanyocr.text_recognizer.mmocr import MMOCR


class MMOCR_Satrn_Sm(MMOCR):
    DICT90 = tuple(
        "0123456789abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()"
        "*+,-./:;<=>?@[\\]_`~"
    )

    def __init__(self, use_gpu: bool = True) -> None:
        model_path = download_model("MMOCR_SATRN_SM.onnx")

        self.idx2char = list(self.DICT90)

        self.start_end_token = "<BOS/EOS>"
        self.unknown_token = "<UKN>"
        self.padding_token = "<PAD>"

        self.idx2char.append(self.unknown_token)
        self.unknown_idx = len(self.idx2char) - 1

        self.idx2char.append(self.start_end_token)
        self.start_idx = len(self.idx2char) - 1
        self.end_idx = len(self.idx2char) - 1

        self.idx2char.append(self.padding_token)
        padding_idx = len(self.idx2char) - 1

        self.ignore_indexes = [padding_idx]

        super().__init__(use_gpu, model_path, 100, 32)

        self.set_max_batch_size(1)

    def normalize_img(self, bgr_img):
        img = cv2.resize(bgr_img, (self.model_w, self.model_h)).astype(np.float32)[
            :, :, ::-1
        ]
        img /= 255

        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406

        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = np.transpose(img, [2, 0, 1])

        return img

    def decode(self, pred) -> Tuple[str, float]:
        result = np.argmax(pred, axis=-1)

        text = ""
        score = 1.0
        for idx, scores in zip(result, pred):
            if idx in self.ignore_indexes:
                continue

            if idx == self.end_idx:
                break

            text += self.idx2char[idx]
            score *= scores[idx]

        return (text, score)

    def set_max_batch_size(self, batch_size: int):
        # this model does not support batching
        self.max_batch_size = 1
