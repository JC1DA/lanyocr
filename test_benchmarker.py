from lanyocr import LanyOcr
from lanyocr.benchmarker.benchmarker_icdar2015 import LanyBenchmarkerICDAR2015

ocr = LanyOcr()
benchmarker = LanyBenchmarkerICDAR2015(ocr, "./datasets/ICDAR/2015")

accuracy = benchmarker.compute_detector_accuracy()
print(accuracy)
