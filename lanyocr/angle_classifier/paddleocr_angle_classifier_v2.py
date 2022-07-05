from typing import List
import cv2
import onnxruntime
import numpy as np
from lanyocr.angle_classifier import LanyOcrAngleClassifier
from lanyocr.lanyocr_utils import download_model


class PaddleOcrAngleClassifierv2(LanyOcrAngleClassifier):
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu)

        self.model_h = 48
        self.model_w = 192

        model_path = download_model("lanyocr-paddleocr-direction-classifier.onnx")

        # load model
        providers = ["CPUExecutionProvider"]
        if self.use_gpu:
            providers = ["CUDAExecutionProvider"]

        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )

        self.session = onnxruntime.InferenceSession(
            model_path, sess_options=opts, providers=providers
        )

    def infer(self, bgr_image, thresh=0.5) -> int:
        normed_img = self._resize_norm_img(bgr_image, [3, self.model_h, self.model_w])
        rotated_probs = self.session.run(None, {"x": [normed_img]})[0][0]

        if rotated_probs[1] > thresh:
            return 180

        return 0

    def infer_batch(self, bgr_images, thresh=0.5) -> List[int]:
        results: List[int] = []

        normed_imgs = []
        for img in bgr_images:
            normed_img = self._resize_norm_img(img, [3, self.model_h, self.model_w])
            normed_imgs.append(normed_img)

        rotated_probs = self.session.run(None, {"x": normed_imgs})[0]

        for prob in rotated_probs:
            rotated_angle = 180 if prob[1] > thresh else 0
            results.append(rotated_angle)

        return results

    # internal functions
    def _resize_norm_img(self, img, model_input_shape, padding=True):
        imgC, imgH, imgW = model_input_shape
        h = img.shape[0]
        w = img.shape[1]

        if h == imgH and w == imgW:
            resized_image = img
        elif not padding:
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=cv2.INTER_LINEAR
            )
            resized_w = imgW
        else:
            ratio = imgH / float(h)
            _w = int(ratio * w)

            _img = cv2.resize(img, (_w, imgH))

            resized_image = np.zeros(shape=(imgH, imgW, 3), dtype=np.uint8)
            if _w > imgW:
                resized_image[:, :, :] = _img[:, :imgW, :]
            else:
                resized_image[:, :_w, :] = _img[:, :, :]

        resized_image = resized_image.astype("float32")
        if model_input_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5

        return resized_image
