#!/bin/bash

mkdir -p models

wget https://lanytek.com/models/lanyocr-easyocr-craft-detector.onnx -O models/lanyocr-easyocr-craft-detector.onnx
wget https://lanytek.com/models/lanyocr-paddleocr-db-recognizer.onnx -O models/lanyocr-paddleocr-db-recognizer.onnx
wget https://lanytek.com/models/lanyocr-paddleocr-direction-classifier.onnx -O models/lanyocr-paddleocr-direction-classifier.onnx