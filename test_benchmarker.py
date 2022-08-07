from lanyocr import LanyOcr
from lanyocr.benchmarker.benchmarker_icdar2015 import LanyBenchmarkerICDAR2015

ocr = LanyOcr()
benchmarker = LanyBenchmarkerICDAR2015(ocr, "./datasets/ICDAR/2015")

detector_accuracy = benchmarker.compute_detector_accuracy()
print(f"DETECTOR: {detector_accuracy}")

recognizer_accuracy = benchmarker.compute_recognizer_accuracy()
print(f"RECOGNIZER: {recognizer_accuracy}")

e2e_accuracy = benchmarker.compute_e2e_accuracy()
print(f"E2E: {e2e_accuracy}")
