import json
import math
import os
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime
from lanyocr.text_recognizer import LanyOcrRecognizer


class PaddleOcrBase(LanyOcrRecognizer):
    def __init__(
        self,
        use_gpu: bool = True,
        model_ignored_tokens: List[int] = [],
        model_characters: List[str] = [],
        accepted_characters: List[str] = [],
        model_path: str = "",
        model_w: int = 320,
        model_h: int = 32,
    ) -> None:
        super().__init__(use_gpu)

        self.model_w = model_w
        self.model_h = model_h
        self.model_ignored_tokens = model_ignored_tokens
        self.model_characters = model_characters
        self.accepted_characters = accepted_characters
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

        preds = self.session.run(None, {"x": [norm_img]})[0][0]

        preds_idx = preds.argmax(axis=1)
        preds_prob = preds.max(axis=1)
        results = self.decode([preds_idx], [preds_prob], True)
        text, prob = results[0]
        return text, prob

    def infer_batch(self, bgr_images) -> List[Tuple[str, float]]:
        batch_results = []
        normed_imgs = [self.normalize_img(img) for img in bgr_images]

        for idx in range(0, len(normed_imgs), self.max_batch_size):
            batch_preds = self.session.run(
                None, {"x": normed_imgs[idx : idx + self.max_batch_size]}
            )[0]

            for preds in batch_preds:
                preds_idx = preds.argmax(axis=1)
                preds_prob = preds.max(axis=1)
                results = self.decode([preds_idx], [preds_prob], True)
                batch_results.append(results[0])

        return batch_results

    def get_model_height(self) -> int:
        return self.model_h

    def get_model_width(self) -> int:
        return self.model_w

    def normalize_img(self, bgr_img):
        h, w = bgr_img.shape[:2]

        if h != self.model_h or w != self.model_w:
            ratio = w / float(h)

            if math.ceil(self.model_h * ratio) > self.model_w:
                resized_w = self.model_w
            else:
                resized_w = int(math.ceil(self.model_h * ratio))

            resized_image = cv2.resize(bgr_img, (resized_w, self.model_h))
        else:
            resized_image = bgr_img

        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5

        if h != self.model_h or w != self.model_w:
            final_img = np.zeros(
                shape=(3, self.model_h, self.model_w), dtype=np.float32
            )
            final_img[:, :, :resized_w] = resized_image
        else:
            final_img = resized_image

        return final_img

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in self.model_ignored_tokens:
                    continue

                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                    ):
                        continue

                detected_char = self.model_characters[int(text_index[batch_idx][idx])]

                # filter out some unused characters
                if (
                    self.accepted_characters
                    and detected_char not in self.accepted_characters
                ):
                    continue

                char_list.append(detected_char)
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            text = "".join(char_list)
            prob = 1.0 if conf_list else 0
            for p in conf_list:
                prob *= p

            result_list.append((text, prob))
        return result_list
