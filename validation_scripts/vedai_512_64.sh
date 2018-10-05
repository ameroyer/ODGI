#!/bin/bash
CUDA_DEVICE=$1


echo '============================================================== FOLD 01'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold01/tiny-yolov2_odgi_512_64/09-27_16-00

echo '============================================================== FOLD 02'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold02/tiny-yolov2_odgi_512_64/09-27_16-00

echo '============================================================== FOLD 03'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold03/tiny-yolov2_odgi_512_64/09-27_16-00

echo '============================================================== FOLD 04'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold04/tiny-yolov2_odgi_512_64/09-27_16-00

echo '============================================================== FOLD 05'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold05/tiny-yolov2_odgi_512_64/09-27_16-00

echo '============================================================== FOLD 06'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold06/tiny-yolov2_odgi_512_64/09-27_16-05

echo '============================================================== FOLD 07'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold07/tiny-yolov2_odgi_512_64/09-27_16-00

echo '============================================================== FOLD 08'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold08/tiny-yolov2_odgi_512_64/09-27_16-05

echo '============================================================== FOLD 09'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold09/tiny-yolov2_odgi_512_64/09-27_16-05

echo '============================================================== FOLD 10'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/vedai_fold10/tiny-yolov2_odgi_512_64/09-27_16-05