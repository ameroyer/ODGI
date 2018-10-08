#!/bin/bash


echo '============================================================== 512x256'
for i in $(seq 1 10); 
do 
    python3 time_model.py --test_num_crops $i --test_patch_nms_threshold 0.25 --test_patch_confidence_threshold 0.1 --test_patch_strong_confidence_threshold 0.9 ./log/vedai_fold01/tiny-yolov2_odgi_512_256/09-28_17-50
done


echo '============================================================== ODGI 512x64'
for i in $(seq 1 10); 
do 
    python3 time_model.py --test_num_crops $i --test_patch_nms_threshold 0.25 --test_patch_confidence_threshold 0.1 --test_patch_strong_confidence_threshold 0.9 ./log/vedai_fold01/tiny-yolov2_odgi_512_64/09-27_16-00
done


echo '============================================================== ODGI 256x128'
for i in $(seq 1 10); 
do 
    python3 time_model.py --test_num_crops $i --test_patch_nms_threshold 0.25 --test_patch_confidence_threshold 0.1 --test_patch_strong_confidence_threshold 0.9 ./log/vedai_fold01/tiny-yolov2_odgi_256_128/09-25_14-48
done


echo '============================================================== ODGI 256x64'
for i in $(seq 1 10); 
do 
    python3 time_model.py --test_num_crops $i --test_patch_nms_threshold 0.25 --test_patch_confidence_threshold 0.1 --test_patch_strong_confidence_threshold 0.9 ./log/vedai_fold01/tiny-yolov2_odgi_256_128/09-23_10-49
done


echo '============================================================== ODGI 128x64'
for i in $(seq 1 10); 
do 
    python3 time_model.py --test_num_crops $i --test_patch_nms_threshold 0.25 --test_patch_confidence_threshold 0.1 --test_patch_strong_confidence_threshold 0.9 ./log/vedai_fold01/tiny-yolov2_odgi_256_128/09-27_10-44
done


echo '============================================================== STANDARD 1024'
python3 time_model.py ./log/vedai_fold01/tiny-yolov2_standard_1024/09-25_13-58/


echo '============================================================== STANDARD 512'
python3 time_model.py ./log/vedai_fold01/tiny-yolov2_standard_512/09-23_11-14/

echo '============================================================== STANDARD 256'
python3 time_model.py ./log/vedai_fold01/tiny-yolov2_standard_512/09-26_12-46/

echo '============================================================== STANDARD 128'
python3 time_model.py ./log/vedai_fold01/tiny-yolov2_standard_512/09-26_18-24/