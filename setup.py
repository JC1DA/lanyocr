import os
from distutils.core import setup

import setuptools

readme = open("README.md", "r").read()

setup(
    name="lanyocr",  # How you named your package folder (MyLib)
    packages=[
        "lanyocr",
        "lanyocr.angle_classifier",
        "lanyocr.benchmarker",
        "lanyocr.text_merger",
        "lanyocr.text_detector",
        "lanyocr.text_recognizer",
        "lanyocr.text_recognizer.dicts",
    ],  # Chose the same as "name"
    version="0.1.2",  # Start with a small number and increase it with every change you make
    license="MIT",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="An OCR library for Python",  # Give a short description about your library
    long_description=readme,
    long_description_content_type="text/markdown",
    author="JC1DA",  # Type in your name
    author_email="jc1da.3011@gmail.com",  # Type in your E-Mail
    url="https://github.com/JC1DA/lanyocr",  # Provide either the link to your github or to your website
    download_url="https://github.com/JC1DA/lanyocr",  # I explain this later on
    keywords=["ocr"],  # Keywords that define your package best
    install_requires=[
        "onnxruntime-gpu>=1.13",
        "numpy>=1.21",
        "opencv-python>=4.5",
        "pydantic>=1.10",
        "shapely",
        "pyclipper",
        "requests",
    ],  # I get to this in a second
    setup_requires=["wheel"],
    python_requires=">=3.0",
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",  # Again, pick a license
        "Programming Language :: Python :: 3.6",  # Specify which pyhton versions that you want to support
    ],
    include_package_data=True,
    data_files=[
        (
            "lanyocr/text_recognizer/dicts",
            [
                "lanyocr/text_recognizer/dicts/paddleocr_en_dict.json",
                "lanyocr/text_recognizer/dicts/paddleocr_french_dict.json",
                "lanyocr/text_recognizer/dicts/paddleocr_latin_dict.json",
                "lanyocr/text_recognizer/dicts/ppocr_keys_v1_mod.json",
            ],
        )
    ],
)
