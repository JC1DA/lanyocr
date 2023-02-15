import argparse

from lanyocr import LanyOcr


def main(args):
    ocr = LanyOcr(
        detector_name=args.detector_name,
        recognizer_name=args.recognizer_name,
        merger_name=args.merger_name,
        merge_boxes_inference=args.merge_boxes_inference,
        merge_rotated_boxes=args.merge_rotated_boxes,
        merge_vertical_boxes=args.merge_vertical_boxes,
        use_gpu=args.use_gpu,
        debug=args.debug,
    )

    results = ocr.infer_from_file(args.image_path)
    ocr.visualize(
        args.image_path,
        results,
        visualize_sub_boxes=False,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LanyOCR")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./outputs/output.jpg")
    parser.add_argument(
        "--detector_name",
        type=str,
        default="paddleocr_en_ppocr_v3",
        help='Name of detector, must be one of these ["easyocr_craft", "paddleocr_en_ppocr_v3"]',
    )
    parser.add_argument(
        "--recognizer_name",
        type=str,
        default="paddleocr_en_ppocr_v3",
        help='Name of recognizer, must be one of these ["paddleocr_en_server", "paddleocr_en_mobile", "paddleocr_en_ppocr_v3", "paddleocr_french_mobile", "paddleocr_latin_mobile"]',
    )
    parser.add_argument(
        "--merger_name",
        type=str,
        default="lanyocr_nomerger",
        help="Name of merger, must be one of these ['lanyocr_nomerger', 'lanyocr_craftbased']",
    )
    parser.add_argument(
        "--merge_rotated_boxes",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="enable merging text boxes in upward/downward directions",
    )
    parser.add_argument(
        "--merge_vertical_boxes",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
        help="enable merging text boxes in vertical direction",
    )
    parser.add_argument(
        "--merge_boxes_inference",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
        help="merge boxes in a text line into images before running inferece to speedup recognition step",
    )
    parser.add_argument(
        "--use_gpu",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
        help="use GPU for inference",
    )
    parser.add_argument(
        "--debug",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
        help="generate text lines and text boxes images for debugging purposes",
    )
    args = parser.parse_args()

    main(args)
