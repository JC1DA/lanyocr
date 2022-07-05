import onnxruntime
import cv2
import math
import numpy as np
from typing import List
from lanyocr.text_detector import LanyOcrDetector
from lanyocr.lanyocr_utils import LanyOcrRRect, download_model


class EasyOcrCraft(LanyOcrDetector):
    def __init__(self, use_gpu: bool = True) -> None:
        super().__init__(use_gpu)

        print("Detector: EasyOcrCraft")

        self.max_square_size = 1536

        model_path = download_model("lanyocr-easyocr-craft-detector.onnx")
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

    def infer(self, bgr_image) -> List[LanyOcrRRect]:
        rrects: List[LanyOcrRRect] = []
        normed_image = self.preprocess(bgr_image)
        det_preds = self.session.run(["output"], {"input": [normed_image]})[0]
        det_polys = self.postprocess(det_preds[0], bgr_image)

        for poly in det_polys:
            cnts = np.array(poly).reshape((1, -1, 2))
            rrect = cv2.minAreaRect(cnts)

            rrects.append(
                LanyOcrRRect(
                    rrect=rrect,
                    points=np.reshape(poly, [-1, 2]).tolist(),
                    direction="",
                )
            )

        return rrects

    # Internal functions

    def load_image(self, image_path: str):
        image = cv2.imread(image_path)
        normed_image = self.preprocess(image)
        return normed_image, image

    def preprocess(self, image):
        img_resized, _, _ = self.resize_aspect_ratio(
            image, square_size=self.max_square_size, interpolation=cv2.INTER_LINEAR
        )
        img_resized = img_resized[:, :, ::-1]
        img_resized = self.normalizeMeanVariance(img_resized)
        img_resized = np.transpose(img_resized, [2, 0, 1])
        return img_resized

    def resize_aspect_ratio(self, img, square_size, interpolation, mag_ratio=1):
        height, width, channel = img.shape

        # magnify image size
        target_size = mag_ratio * max(height, width)

        # set original image size
        if target_size > square_size:
            target_size = square_size

        ratio = target_size / max(height, width)

        target_h, target_w = int(height * ratio), int(width * ratio)
        proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

        # make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resized[0:target_h, 0:target_w, :] = proc
        target_h, target_w = target_h32, target_w32

        size_heatmap = (int(target_w / 2), int(target_h / 2))

        return resized, ratio, size_heatmap

    def normalizeMeanVariance(
        self, in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
    ):
        # should be RGB order
        img = in_img.copy().astype(np.float32)

        img -= np.array(
            [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
        )
        img /= np.array(
            [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
            dtype=np.float32,
        )
        return img

    # ref: https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/detection.py
    def postprocess(
        self,
        preds,
        original_img,
        max_img_size: int = 1536,
        text_threshold: float = 0.7,
        low_text_threshold: float = 0.33,
        # configs to detect more texts
        # text_threshold: float = 0.6,
        # low_text_threshold: float = 0.25,
        link_text_threshold: float = 0.125,
    ):
        height, width = original_img.shape[:2]

        target_size = max(height, width)
        if target_size > max_img_size:
            target_size = max_img_size

        target_ratio = float(target_size) / max(height, width)
        ratio_h = ratio_w = 1.0 / target_ratio

        score_text = preds[:, :, 0]
        score_link = preds[:, :, 1]

        # Post-processing
        estimate_num_chars = False
        boxes, polys, mapper = self.getDetBoxes(
            score_text,
            score_link,
            text_threshold=text_threshold,
            link_threshold=link_text_threshold,
            low_text=low_text_threshold,
            poly=False,
            estimate_num_chars=estimate_num_chars,
        )

        # coordinate adjustment
        boxes = self.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = self.adjustResultCoordinates(polys, ratio_w, ratio_h)

        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        result = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            result.append(poly)

        return result

    # ref: https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/craft_utils.py
    def getDetBoxes(
        self,
        textmap,
        linkmap,
        text_threshold,
        link_threshold,
        low_text,
        poly=False,
        estimate_num_chars=False,
    ):
        if poly and estimate_num_chars:
            raise Exception(
                "Estimating the number of characters not currently supported with poly."
            )
        boxes, labels, mapper = self.getDetBoxes_core(
            textmap,
            linkmap,
            text_threshold,
            link_threshold,
            low_text,
            estimate_num_chars,
        )

        if poly:
            # polys = getPoly_core(boxes, labels, mapper, linkmap)
            pass
        else:
            polys = [None] * len(boxes)

        return boxes, polys, mapper

    # ref: https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/craft_utils.py

    def getDetBoxes_core(
        self,
        textmap,
        linkmap,
        text_threshold,
        link_threshold,
        low_text,
        estimate_num_chars=False,
    ):
        # prepare data
        linkmap = linkmap.copy()
        textmap = textmap.copy()
        img_h, img_w = textmap.shape

        """ labeling method """
        ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
        ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

        text_score_comb = np.clip(text_score + link_score, 0, 1)
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            text_score_comb.astype(np.uint8), connectivity=4
        )

        det = []
        mapper = []
        for k in range(1, nLabels):
            # size filtering
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10:
                continue

            # thresholding
            if np.max(textmap[labels == k]) < text_threshold:
                continue

            # make segmentation map
            segmap = np.zeros(textmap.shape, dtype=np.uint8)
            segmap[labels == k] = 255
            mapper.append(k)
            segmap[
                np.logical_and(link_score == 1, text_score == 0)
            ] = 0  # remove link area
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            # boundary check
            if sx < 0:
                sx = 0
            if sy < 0:
                sy = 0
            if ex >= img_w:
                ex = img_w
            if ey >= img_h:
                ey = img_h
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

            # make box
            np_contours = (
                np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
                .transpose()
                .reshape(-1, 2)
            )
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)

            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
                t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # make clock-wise order
            startidx = box.sum(axis=1).argmin()
            box = np.roll(box, 4 - startidx, 0)
            box = np.array(box)

            det.append(box)

        return det, labels, mapper

    # ref: https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/craft_utils.py
    def adjustResultCoordinates(self, polys, ratio_w, ratio_h, ratio_net=2):
        if len(polys) > 0:
            polys = np.array(polys)
            for k in range(len(polys)):
                if polys[k] is not None:
                    polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
        return polys
