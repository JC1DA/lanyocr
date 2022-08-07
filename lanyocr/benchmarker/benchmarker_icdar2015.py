import os
from glob import glob

from lanyocr.benchmarker import LanyBenchmarker


class LanyBenchmarkerICDAR2015(LanyBenchmarker):
    def load_dataset(self):
        image_paths_dir = os.path.join(self.dataset_path, "images")
        gt_paths_dir = os.path.join(self.dataset_path, "validation")

        image_paths = glob(image_paths_dir + "/*")
        gt_paths = glob(gt_paths_dir + "/*")

        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            image_id = int(image_name.split(".")[0].split("_")[-1])
            self.id2imagePath[image_id] = image_path

        for gt_path in gt_paths:
            gt_name = os.path.basename(gt_path)
            gt_id = int(gt_name.split(".")[0].split("_")[-1])
            self.id2gts[gt_id] = (
                open(gt_path, "r", encoding="utf-8-sig").read().encode()
            )

        assert (
            len(self.id2imagePath) != 0
        ), "Make sure dataset_path contains images and validation directories"
