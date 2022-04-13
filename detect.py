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
    # providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
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
    t0 = time.time()
    det_preds = detector_session.run(["output"], {"input": imgs})[0]
    t1 = time.time()
    det_polys = postprocess(det_preds[0], img)
    t2 = time.time()
    det_lines = merge_text_boxes(det_polys, merge_vertical=args.merge_vertical)
    t3 = time.time()

    original_img = np.array(img)
    vis_img = np.array(original_img)

    rec_lat = 0
    extra_rec_lat = 0

    direction_lat = 0
    extra_direction_lat = 0

    COLORS = [
        # (0, 0, 0),
        (127, 0, 0),
        (255, 0, 0),
        (0, 127, 0),
        (0, 255, 0),
        (0, 0, 127),
        (0, 0, 255),
        (127, 127, 0),
        (255, 255, 0),
        (0, 127, 127),
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

            # if is_flipped(direction_classifier_session, padded_img, 0.925):
            #     flipped_count += 1

        for flipped in is_flipped_batch(
            direction_classifier_session, padded_imgs, 0.925
        ):
            if flipped:
                flipped_count += 1

        direction_lat += time.time() - direction_lat_t0

        if flipped_count > num_tries // 2:
            line_img = cv2.rotate(line_img, cv2.cv2.ROTATE_180)

        # cv2.imwrite(f"./tmp/line_{line_idx}.jpg", line_img)

        # recognize each text independently for best accuracy
        for sub_rrect in line.sub_rrects:
            sub_rrect: LanyOcrRRect = sub_rrect

            _img = np.array(original_img)
            dst_img = crop_rrect(_img, sub_rrect.rrect)

            # use template matching with the text line box to determine the correct orientation of the box
            _scores = []
            _imgs = []

            rotated_img = dst_img
            for _ in range(4):
                max_val = 0

                if (
                    rotated_img.shape[0] <= line_img.shape[0]
                    and rotated_img.shape[1] <= line_img.shape[1]
                ):
                    res = cv2.matchTemplate(line_img, rotated_img, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)

                _scores.append(max_val)
                _imgs.append(rotated_img)

                rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_CLOCKWISE)

            best_idx = np.argmax(_scores)
            best_score = _scores[best_idx]
            best_img = _imgs[best_idx]

            _scores[best_idx] = 0
            second_best_idx = np.argmax(_scores)
            second_best_score = _scores[second_best_idx]
            second_best_img = _imgs[second_best_idx]

            dst_img = _imgs[best_idx]

            # in case we don't have clear answers from template matching, use direction model to determine
            if best_score - second_best_score < 0.1:
                extra_direction_lat_t0 = time.time()
                if not is_flipped(direction_classifier_session, best_img, 0.5):
                    dst_img = best_img
                elif not is_flipped(direction_classifier_session, second_best_img, 0.5):
                    dst_img = second_best_img
                extra_direction_lat += time.time() - extra_direction_lat_t0

            sub_rrect_h, sub_rrect_w = dst_img.shape[:2]
            sub_rrect_ratio = float(sub_rrect_h) / sub_rrect_w

            # sometimes, if small text box is detected as a square/rectangle with 0 or 90 degrees angle, we have to rotate the box relative to the text line to get horizontal text
            if sub_rrect_ratio >= 0.75 and sub_rrect_ratio <= 1.25:
                sub_rrect_angle = sub_rrect.rrect[2]

                if abs(line_angle - sub_rrect_angle) > 10:
                    dst_img = rotate_image(dst_img, line_angle)

            rec_t0 = time.time()
            sub_text, sub_prob = recognize_text(recognizer_session, dst_img)
            rec_lat += time.time() - rec_t0

            if not isascii(sub_text):
                extra_rec_t0 = time.time()
                for angle in [45, 135]:
                    _dst_img = rotate_image(dst_img, angle)
                    sub_text, sub_prob = recognize_text(recognizer_session, _dst_img)
                    if isascii(sub_text):
                        dst_img = _dst_img
                        break
                extra_rec_lat += time.time() - extra_rec_t0

            if sub_text != "":
                if text != "":
                    text += " "
                text += sub_text
                prob *= sub_prob

        texts.append(text)

        points = cv2.boxPoints(line.get_rrect())
        box = np.int0(np.array(points))
        x_center = np.sum(box[:, 0]) // 4
        y_center = np.sum(box[:, 1]) // 4
        # vis_img = cv2.drawContours(vis_img, [box], 0, (0, 255, 0), 1)
        vis_img = cv2.drawContours(vis_img, [box], 0, COLORS[line_idx % len(COLORS)], 2)

        vis_img = cv2.putText(
            vis_img,
            # f"{text}",
            f"{line_idx}",
            (int(x_center), int(y_center)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        print(f"Line {line_idx}: {text}")

    t4 = time.time()

    cv2.imwrite("outputs/output.jpg", vis_img)

    print("Detector latency: {:.4f} secs".format(t1 - t0))
    print("Detector postproc latency: {:.4f} secs".format(t2 - t1))
    print("Detector merge latency: {:.4f} secs".format(t3 - t2))
    print("Direction latency: {:.4f} secs".format(direction_lat))
    print("Direction extra latency: {:.4f} secs".format(extra_direction_lat))
    print("Recognizer latency: {:.4f} secs".format(rec_lat))
    print("Recognizer extra latency: {:.4f} secs".format(extra_rec_lat))
    print("Recognizer total latency: {:.4f} secs".format(t4 - t3))
    print("E2e latency: {:.4f} secs".format(t4 - t0))


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
    args = parser.parse_args()

    main(args)
