import pickle
import json
import os
import time
import onnxruntime
import argparse
import shutil
from utils import *
from det_utils import (
    load_img,
    postprocess,
    merge_text_boxes,
)
from rec_utils import *
from direction_utils import *

ROTATE_90DEGREES_THRESH = 1.5


def main(args):
    # load models
    providers = ["CUDAExecutionProvider"]
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    detector_session = onnxruntime.InferenceSession(
        args.craft_detector_path, sess_options=opts, providers=providers
    )

    direction_classifier_session = onnxruntime.InferenceSession(
        args.paddle_direction_classifier_path, sess_options=opts, providers=providers
    )
    recognizer_session = onnxruntime.InferenceSession(
        args.paddle_recognizer_path, sess_options=opts, providers=providers
    )

    # load image
    resized_img, img = load_img(args.image_path)
    imgs = [resized_img]

    # run detector
    det_preds = detector_session.run(["output"], {"input": imgs})[0]
    det_polys = postprocess(det_preds[0], img)
    det_lines = merge_text_boxes(det_polys, merge_vertical=args.merge_vertical)

    original_img = np.array(img)
    vis_img = np.array(original_img)

    COLORS = [
        # (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
    ]

    texts = []
    for line_idx, line in enumerate(det_lines):
        text = ""
        prob = 1.0

        line_rrect = line.get_rrect()
        line_angle = line_rrect[2]

        line_img = crop_rrect(np.array(original_img), line_rrect)

        line_h, line_w = line_img.shape[:2]
        if line_h > ROTATE_90DEGREES_THRESH * line_w:
            line_img = cv2.rotate(line_img, cv2.ROTATE_90_CLOCKWISE)

        # determine if a text is 180 degrees flipped or not
        flipped_count = 0
        num_tries = 3
        line_flipped = False

        direction_lat_t0 = time.time()
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
        for flipped in is_flipped_batch(
            direction_classifier_session, padded_imgs, 0.925
        ):
            if flipped:
                flipped_count += 1

        if flipped_count > num_tries // 2:
            line_img = cv2.rotate(line_img, cv2.cv2.ROTATE_180)
            line_flipped = True

        merged_sub_rrect_list: List[List[LanyOcrRRect]] = []
        merged_imgs: List = []
        spacing = 10

        merged_img = np.ones(shape=(32, 320, 3), dtype=np.uint8) * 255
        s = 0

        for sub_rrect_idx, sub_rrect in enumerate(line.sub_rrects):
            if len(merged_sub_rrect_list) == 0:
                merged_sub_rrect_list.append([])

            sub_rrect: LanyOcrRRect = sub_rrect
            rrect_center = sub_rrect.rrect[0]

            # rotate sub-rrect to horizontal line
            delta_x = abs(line_rrect[0][0] - rrect_center[0])
            delta_y = abs(line_rrect[0][1] - rrect_center[1])
            d = distance(line_rrect[0], rrect_center)

            if delta_x > delta_y:
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
            w, h = sub_rrect.rrect[1]

            new_x, new_y, w, h = int(new_x), int(new_y), int(w), int(h)

            if h > w:
                w, h = h, w

            left = max(0, new_x - w // 2)
            right = min(new_x + w // 2, line_img.shape[1] - 1)

            dst_img = line_img[:, left:right, :]

            sub_rrect_h, sub_rrect_w = dst_img.shape[:2]
            sub_rrect_ratio = float(sub_rrect_h) / sub_rrect_w

            # sometimes, if small text box is detected as a square/rectangle with 0 or 90 degrees angle, we have to rotate the box relative to the text line to get horizontal text
            if sub_rrect_ratio >= 0.75 and sub_rrect_ratio <= 1.25:
                sub_rrect_angle = sub_rrect.rrect[2]

                if abs(line_angle - sub_rrect_angle) > 10:
                    dst_img = rotate_image(dst_img, line_angle)

            resized_dst_img = resize_img_to_height(dst_img, 32)
            h, w = resized_dst_img.shape[:2]

            if s + w >= args.text_recognizer_width - 5:
                # do not merge sub-rrect into single image anymore
                merged_img[:, s + 1 :, :] = 127
                merged_imgs.append(merged_img)
                # initialize new merge
                merged_img = np.ones(shape=(32, 320, 3), dtype=np.uint8) * 255
                s = 0
                merged_sub_rrect_list.append([])

            merged_sub_rrect_list[-1].append(sub_rrect)

            if w <= args.text_recognizer_width:
                merged_img[:, s : s + w, :] = resized_dst_img
            else:
                # very long sub-rrect
                resized_dst_img = resize_img_to_width(resized_dst_img, 320)
                h, w = resized_dst_img.shape[:2]
                merged_img[:h, :, :] = resized_dst_img

            s += w

            if args.merge_boxes_inference:
                # pick a color to set as background color between text-words
                col1 = merged_img[2:4, :s, :]
                col2 = merged_img[28:30, :s, :]
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

                padding_value = np.array([b, g, r], dtype=np.uint8)
                merged_img[:, s : s + spacing, :] = padding_value
                s += spacing

            if (
                not args.merge_boxes_inference
                or sub_rrect_idx == len(line.sub_rrects) - 1
            ):
                merged_img[:, s + 1 :, :] = 127
                merged_imgs.append(merged_img)
                merged_img = np.ones(shape=(32, 320, 3), dtype=np.uint8) * 255
                s = 0

                if sub_rrect_idx < len(line.sub_rrects) - 1:
                    merged_sub_rrect_list.append([])

        recogition_results = recognize_text_batch(recognizer_session, merged_imgs)

        for sub_rrects, recognition_result in zip(
            merged_sub_rrect_list, recogition_results
        ):
            sub_text, sub_prob = recognition_result

            if sub_text != "":
                if text != "":
                    text += " "
                text += sub_text
                prob *= sub_prob

        # visualize boxes and texts
        points = cv2.boxPoints(line.get_rrect())
        box = np.int0(np.array(points))
        x_center = np.sum(box[:, 0]) // 4
        y_center = np.sum(box[:, 1]) // 4

        vis_img = cv2.drawContours(vis_img, [box], 0, COLORS[line_idx % len(COLORS)], 2)

        vis_img = cv2.putText(
            vis_img,
            f"{text}",
            # f"{line_idx}",
            (int(x_center), int(y_center)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            COLORS[line_idx % len(COLORS)],
            1,
            cv2.LINE_AA,
        )

        print(f"Line {line_idx}: {text}")

    cv2.imwrite("outputs/output.jpg", vis_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nxs")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument(
        "--craft_detector_path",
        type=str,
        default="models/lanyocr-easyocr-craft-detector.onnx",
    )
    parser.add_argument(
        "--paddle_recognizer_path",
        type=str,
        default="models/lanyocr-paddleocr-db-recognizer.onnx",
    )
    parser.add_argument(
        "--paddle_direction_classifier_path",
        type=str,
        default="models/lanyocr-paddleocr-direction-classifier.onnx",
    )
    parser.add_argument(
        "--merge_vertical", default=False, type=lambda x: (str(x).lower() == "true")
    )
    parser.add_argument(
        "--merge_boxes_inference",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument("--text_recognizer_height", type=int, default=32)
    parser.add_argument("--text_recognizer_width", type=int, default=320)
    args = parser.parse_args()

    main(args)
