#!/bin/bash
CUDA_DEVICE=1


echo '============================================================== FOLD 01'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold01/tiny-yolov2_odgi_128_64/09-27_10-44

echo '============================================================== FOLD 02'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold02/tiny-yolov2_odgi_128_64/09-27_10-44

echo '============================================================== FOLD 03'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold03/tiny-yolov2_odgi_128_64/09-28_10-21

echo '============================================================== FOLD 04'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold04/tiny-yolov2_odgi_128_64/09-28_10-21

echo '============================================================== FOLD 05'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold05/tiny-yolov2_odgi_128_64/09-27_10-44

echo '============================================================== FOLD 06'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold06/tiny-yolov2_odgi_128_64/09-27_10-44

echo '============================================================== FOLD 07'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold07/tiny-yolov2_odgi_128_64/09-27_13-31

echo '============================================================== FOLD 08'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold08/tiny-yolov2_odgi_128_64/09-27_14-09

echo '============================================================== FOLD 09'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold09/tiny-yolov2_odgi_128_64/09-27_14-09

echo '============================================================== FOLD 10'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold10/tiny-yolov2_odgi_128_64/09-27_14-09