#!/bin/bash
rm -rf dist
python setup.py sdist bdist_wheel
rm -rf build
rm -rf *.egg-info
