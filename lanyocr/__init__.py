import math
import os
from typing import List

import cv2
import numpy as np

from lanyocr.angle_classifier import LanyOcrAngleClassifierFactory
from lanyocr.lanyocr_utils import LanyOcrResult
from lanyocr.lanyocr_utils import LanyOcrRRect
from lanyocr.lanyocr_utils import LanyOcrTextLine
from lanyocr.lanyocr_utils import crop_rrect
from lanyocr.lanyocr_utils import distance
from lanyocr.lanyocr_utils import resize_img_to_height
from lanyocr.lanyocr_utils import resize_img_to_width
from lanyocr.lanyocr_utils import rotate_image
from lanyocr.lanyocr_utils import rotate_image_at_center
from lanyocr.text_detector import LanyOcrDetectorFactory
from lanyocr.text_merger import LanyOcrMergerFactory
from lanyocr.text_recognizer import LanyOcrRecognizerFactory


class LanyOcr:
    def __init__(
        self,
        detector_name: str = "paddleocr_en_ppocr_v3",
        recognizer_name: str = "paddleocr_en_ppocr_v3",
        angle_classifier_name: str = "paddleocr_mobile",
        merger_name: str = "lanyocr_nomerger",
        merge_boxes_inference: bool = False,
        merge_rotated_boxes: bool = True,
        merge_vertical_boxes: bool = False,
        use_gpu: bool = False,
        debug: bool = False,
    ) -> None:
        if merge_boxes_inference and recognizer_name in [
            "mmocr_satrn",
            "mmocr_satrn_sm",
            "paddleocr_en_ppocr_v3",
        ]:
            merge_boxes_inference = False
            if recognizer_name == "paddleocr_en_ppocr_v3":
                print(
                    f"Disabled merge_boxes_inference because {recognizer_name} has dynamic input width."
                )
            else:
                print(
                    f"Disabled merge_boxes_inference because {recognizer_name} could not recognize space character."
                )

        self.detector = LanyOcrDetectorFactory.create(detector_name, use_gpu=use_gpu)
        self.recognizer = LanyOcrRecognizerFactory.create(
            recognizer_name, use_gpu=use_gpu
        )
        self.angle_classifier = LanyOcrAngleClassifierFactory.create(
            angle_classifier_name, use_gpu=use_gpu
        )
        self.merger = LanyOcrMergerFactory.create(merger_name)

        self.rotate_90degrees_thresh = 1.5
        self.spacing = 10
        self.merge_boxes_inference = merge_boxes_inference
        self.merge_rotated_boxes = merge_rotated_boxes
        self.merge_vertical_boxes = merge_vertical_boxes

        self.debug: bool = debug
        self.debug_dir = "./debug"

        if self.debug:
            import shutil

            if os.path.exists(self.debug_dir):
                shutil.rmtree(self.debug_dir)
            os.makedirs(self.debug_dir)

    def infer_from_file(self, image_path: str) -> List[LanyOcrResult]:
        image = cv2.imread(image_path)
        return self.infer(image)

    def infer(self, brg_image) -> List[LanyOcrResult]:
        ocr_results: List[LanyOcrResult] = []

        det_rrects = self.detector.infer(brg_image)
        det_lines = self.merger.merge_to_lines(
            det_rrects, self.merge_rotated_boxes, self.merge_vertical_boxes
        )

        original_img = np.array(brg_image)

        for line_idx, line in enumerate(det_lines):
            result = self._infer_line(original_img, line, line_idx)
            if result.text != "":
                ocr_results.append(result)

            if self.debug:
                print(f'Line {line_idx} - Text: "{result.text}" - Score: {result.prob}')

        return ocr_results

    def visualize(
        self,
        image,
        results: List[LanyOcrResult],
        visualize_sub_boxes: bool = False,
        output_path: str = "outputs/output.jpg",
        font_scale: float = 0.5,
    ):
        print("Visualizing results...")

        COLORS = [
            # (0, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            # (0, 255, 255),
        ]

        if isinstance(image, str):
            image = cv2.imread(image)

        vis_img = np.ones(image.shape, dtype=np.uint8) * 255

        for idx, result in enumerate(results):
            line_rrect = result.line.get_rrect()
            line_angle = line_rrect[2]

            if abs(line_angle) > 89:
                line_angle = 90

            if line_angle == 0 and line_rrect[1][0] * 1.3 < line_rrect[1][1]:
                line_angle = 90
            elif line_angle == 90 and line_rrect[1][1] > 1.3 * line_rrect[1][0]:
                line_angle = 0

            if line_angle not in [0, 90]:
                points = cv2.boxPoints(line_rrect)
                points = np.array(points).reshape((1, -1, 2))
                sorted_cnts = points[0][points[0][:, 0].argsort()[::-1]]
                lowestPointIdx = np.argmax(sorted_cnts[:, 1])
                num_pnt_on_left = points.shape[1] - lowestPointIdx - 1
                if num_pnt_on_left < 2:
                    if line_angle < 45:
                        line_angle = -line_angle
                    else:
                        line_angle = -(90 - line_angle)
                else:
                    line_angle = line_angle

            points = cv2.boxPoints(line_rrect)
            box = np.int0(np.array(points))
            center_x = np.sum(box[:, 0]) // 4
            center_y = np.sum(box[:, 1]) // 4

            vis_img = cv2.drawContours(vis_img, [box], 0, COLORS[idx % len(COLORS)], 2)
            image = cv2.drawContours(image, [box], 0, COLORS[idx % len(COLORS)], 2)

            t_size = cv2.getTextSize(result.text, 0, fontScale=font_scale, thickness=1)[
                0
            ]

            center_x = int(center_x)
            center_y = int(center_y)

            padding_x = max(center_x, image.shape[1] - center_x)
            padding_y = max(center_y, image.shape[0] - center_y)
            padding_size = max(padding_x, padding_y) + 1

            vis_img = cv2.copyMakeBorder(
                vis_img,
                padding_size,
                padding_size,
                padding_size,
                padding_size,
                cv2.BORDER_CONSTANT,
            )

            vis_img = rotate_image_at_center(
                vis_img,
                (center_x + padding_size, center_y + padding_size),
                line_angle,
            )

            vis_img = cv2.putText(
                vis_img,
                f"{result.text}",
                (
                    int(center_x + padding_size - t_size[0] // 2),
                    int(center_y + padding_size + t_size[1] // 2),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            image = cv2.putText(
                image,
                f"{idx}",
                (
                    int(center_x),
                    int(center_y),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            vis_img = rotate_image_at_center(
                vis_img,
                (center_x + padding_size, center_y + padding_size),
                -line_angle,
            )

            vis_img = vis_img[
                padding_size : padding_size + image.shape[0],
                padding_size : padding_size + image.shape[1],
                :,
            ]

            if visualize_sub_boxes:
                for sub_rrect in result.line.sub_rrects:
                    points = sub_rrect.points
                    box = np.int0(np.array(points))
                    vis_img = cv2.drawContours(
                        vis_img, [box], 0, COLORS[(idx + 1) % len(COLORS)], 1
                    )

        if image.shape[1] >= 1.5 * image.shape[0]:
            vis_img = np.concatenate([image, vis_img], axis=0)
        else:
            vis_img = np.concatenate([image, vis_img], axis=1)

        cv2.imwrite(output_path, vis_img)

        print(f"Stored output image in {output_path}")

    def _infer_line(
        self, original_img, line: LanyOcrTextLine, line_idx: int
    ) -> LanyOcrResult:
        text = ""
        prob = 1.0

        line_rrect = line.get_rrect()

        line_img = crop_rrect(np.array(original_img), line_rrect)

        line_h, line_w = line_img.shape[:2]
        if line_h > self.rotate_90degrees_thresh * line_w:
            line_img = cv2.rotate(line_img, cv2.ROTATE_90_CLOCKWISE)

        # determine if a text is 180 degrees flipped or not
        line_flipped = self._is_line_flipped(line_img, 0.925)
        if line_flipped:
            line_img = cv2.rotate(line_img, cv2.ROTATE_180)

        merged_sub_rrect_list: List[List[LanyOcrRRect]] = []
        merged_imgs: List = []

        merged_img = (
            np.ones(
                shape=(
                    self.recognizer.get_model_height(),
                    self.recognizer.get_model_width(),
                    3,
                ),
                dtype=np.uint8,
            )
            * 255
        )
        s = 0

        for sub_rrect_idx, sub_rrect in enumerate(line.sub_rrects):
            if len(merged_sub_rrect_list) == 0:
                merged_sub_rrect_list.append([])

            dst_img = self._crop_and_rotate_rrect_horizontal(
                line_img, line_rrect, sub_rrect, line_flipped
            )

            if not self.merge_boxes_inference:
                # inference on each individual box for best accuracy
                merged_sub_rrect_list[-1].append(sub_rrect)
                merged_imgs.append(dst_img)
                if sub_rrect_idx < len(line.sub_rrects) - 1:
                    merged_sub_rrect_list.append([])
                continue

            resized_dst_img = resize_img_to_height(dst_img, 32)
            h, w = resized_dst_img.shape[:2]

            if s + w >= self.recognizer.get_model_width() - 5:
                # do not merge sub-rrect into single image anymore
                merged_img[:, s + 1 :, :] = 127
                merged_imgs.append(merged_img)
                # initialize new merge
                merged_img = (
                    np.ones(
                        shape=(
                            self.recognizer.get_model_height(),
                            self.recognizer.get_model_width(),
                            3,
                        ),
                        dtype=np.uint8,
                    )
                    * 255
                )
                s = 0
                merged_sub_rrect_list.append([])

            merged_sub_rrect_list[-1].append(sub_rrect)

            if w <= self.recognizer.get_model_width():
                merged_img[:, s : s + w, :] = resized_dst_img
            else:
                # very long sub-rrect
                resized_dst_img = resize_img_to_width(
                    resized_dst_img, self.recognizer.get_model_width()
                )
                h, w = resized_dst_img.shape[:2]
                merged_img[:h, :, :] = resized_dst_img

            s += w

            # pad background color
            merged_img[:, s : s + self.spacing, :] = self._get_padding_color(
                merged_img, s
            )
            s += self.spacing

            if sub_rrect_idx == len(line.sub_rrects) - 1:
                merged_img[:, s + 1 :, :] = 127
                merged_imgs.append(merged_img)
                merged_img = (
                    np.ones(
                        shape=(
                            self.recognizer.get_model_height(),
                            self.recognizer.get_model_width(),
                            3,
                        ),
                        dtype=np.uint8,
                    )
                    * 255
                )
                s = 0

        if self.debug:
            lines_dir = os.path.join(self.debug_dir, "detector/lines")
            if not os.path.exists(lines_dir):
                os.makedirs(lines_dir)

            cv2.imwrite(os.path.join(lines_dir, f"{line_idx}.jpg"), line_img)

            sub_rrect_dir = os.path.join(self.debug_dir, "detector/sub_rrects")
            if not os.path.exists(sub_rrect_dir):
                os.makedirs(sub_rrect_dir)

            for merged_img_idx, merged_img in enumerate(merged_imgs):
                image_path = os.path.join(
                    sub_rrect_dir, f"{line_idx}_{merged_img_idx}.jpg"
                )
                cv2.imwrite(image_path, merged_img)

        recogition_results = self.recognizer.infer_batch(merged_imgs)

        i = 0
        for sub_rrects, recognition_result in zip(
            merged_sub_rrect_list, recogition_results
        ):
            sub_text, sub_prob = recognition_result

            if sub_text == "" and not self.merge_boxes_inference:
                # should we check if this text is 45 degrees rotated
                # NOTE: only works when we do inference on each box independently
                sub_texts = []
                sub_probs = []
                for angle in [45, 135]:
                    rotated_img = rotate_image(merged_imgs[i], angle)
                    extra_result = self.recognizer.infer(rotated_img)
                    sub_texts.append(extra_result[0])
                    sub_probs.append(extra_result[1])
                sub_text = sub_texts[np.argmax(sub_probs)]

            if self.debug:
                print(f"Line {line_idx} - MergedImageIdx {i} - {sub_text} - {sub_prob}")

            if sub_text != "":
                if text != "":
                    text += " "
                text += sub_text
                prob *= sub_prob

            if not self.merge_boxes_inference:
                # sub_rrects should have only 1 object
                sub_rrects[0].text = sub_text

            i += 1

        return LanyOcrResult(text=text, prob=prob, line=line)

    def _get_padding_color(self, merged_img, s):
        # pick a color to set as background color between text-words
        margin = 2
        width = 2

        col1 = merged_img[margin : margin + width, :s, :]
        col2 = merged_img[
            self.recognizer.get_model_height()
            - margin
            - width : self.recognizer.get_model_height()
            - margin,
            :s,
            :,
        ]
        last_column = np.concatenate([col1, col2], axis=0)
        last_column = np.reshape(last_column, [-1, 3])

        b = last_column[:, 0]
        g = last_column[:, 1]
        r = last_column[:, 2]
        b = np.sort(b)
        g = np.sort(g)
        r = np.sort(r)
        b = b[len(b) // 2]
        g = g[len(g) // 2]
        r = r[len(r) // 2]

        return np.array([b, g, r], dtype=np.uint8)

    def _crop_and_rotate_rrect_horizontal(
        self, line_img, line_rrect, sub_lanyrrect: LanyOcrRRect, line_flipped: bool
    ):
        rrect_center = sub_lanyrrect.rrect[0]

        # rotate sub-rrect to horizontal line
        delta_x = abs(line_rrect[0][0] - rrect_center[0])
        delta_y = abs(line_rrect[0][1] - rrect_center[1])
        d = distance(line_rrect[0], rrect_center)

        # print(delta_x, delta_y, d)

        if 1.05 * delta_x > delta_y:
            if rrect_center[0] < line_rrect[0][0]:
                new_x = line_img.shape[1] // 2 - d
            else:
                new_x = line_img.shape[1] // 2 + d
        else:
            if rrect_center[1] < line_rrect[0][1]:
                new_x = line_img.shape[1] // 2 + d
            else:
                new_x = line_img.shape[1] // 2 - d

        if line_flipped:
            new_x = line_img.shape[1] - new_x

        new_y = line_img.shape[0] // 2
        w, h = sub_lanyrrect.rrect[1]

        new_x, new_y, w, h = int(new_x), int(new_y), int(w), int(h)

        if h > w:
            w, h = h, w

        left = max(0, new_x - w // 2)
        right = min(new_x + w // 2, line_img.shape[1] - 1)

        return line_img[:, left:right, :]

    def _is_line_flipped(self, line_img, thresh):
        # determine if a text is 180 degrees flipped or not
        flipped_count = 0
        num_tries = 3

        padded_imgs = []
        for padding_size in range(num_tries):
            padded_img = cv2.copyMakeBorder(
                line_img,
                padding_size,
                padding_size,
                padding_size,
                padding_size,
                cv2.BORDER_CONSTANT,
            )

            padded_imgs.append(padded_img)

        # run in batch
        for rotated_angle in self.angle_classifier.infer_batch(padded_imgs, thresh):
            if rotated_angle == 180:
                flipped_count += 1

        return flipped_count > num_tries // 2
