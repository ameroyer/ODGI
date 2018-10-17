#!/bin/bash

echo '============================================================== STANDARD 1024'
python3 time_model.py --verbose 0 ./log/sdd/yolov2_standard_5boxes_1024/10-11_16-41

echo '============================================================== STANDARD 512'
python3 time_model.py --verbose 0 ./log/sdd/yolov2_standard_5boxes_512/10-11_21-03

echo '============================================================== STANDARD 256'
python3 time_model.py --verbose 0 ./log/sdd/yolov2_standard_5boxes_256/10-11_16-41

echo '============================================================== STANDARD 128'
python3 time_model.py --verbose 0 ./log/sdd/yolov2_standard_5boxes_128/10-12_08-26