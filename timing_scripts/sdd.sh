#!/bin/bash


echo '============================================================== ODI YOLO 512x256'
for i in $(seq 1 10); 
do 
    python3 time_model.py --test_num_crops $i --test_patch_nms_threshold 0.5 --test_patch_confidence_threshold 0.1 --test_patch_strong_confidence_threshold 0.65 --verbose 0 ./log/sdd/yolov2_odgi_512_256/10-05_14-39 
done

echo '============================================================== ODGI YOLO 256x128'
for i in $(seq 1 10); 
do 
    python3 time_model.py --test_num_crops $i --test_patch_nms_threshold 0.5 --test_patch_confidence_threshold 0.1 --test_patch_strong_confidence_threshold 0.65 --verbose 0 ./log/sdd/yolov2_odgi_256_128/10-06_23-45
done

echo '============================================================== ODGI YOLO 128x64'
for i in $(seq 1 10); 
do 
    python3 time_model.py --test_num_crops $i --test_patch_nms_threshold 0.25 --test_patch_confidence_threshold 0.1 --test_patch_strong_confidence_threshold 0.65 --verbose 0 ./log/sdd/yolov2_odgi_128_64/10-06_23-46
done

echo '============================================================== STANDARD 1024'
python3 time_model.py --verbose 0 ./log/sdd/yolov2_standard_1024/10-04_10-19

echo '============================================================== STANDARD 512'
python3 time_model.py --verbose 0 ./log/sdd/yolov2_standard_512/10-03_16-16

echo '============================================================== STANDARD 256'
python3 time_model.py --verbose 0 ./log/sdd/yolov2_standard_256/10-04_13-18

echo '============================================================== STANDARD 128'
python3 time_model.py --verbose 0 ./log/sdd/yolov2_standard_128/10-04_13-18