#!/bin/bash
CUDA_DEVICE=$1


echo '============================================================== SDD 512x256'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/sdd/tiny-yolov2_odgi_512_256/10-04_10-18

echo '============================================================== SDD 512x64'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/sdd/tiny-yolov2_odgi_512_64/10-04_10-18

echo '============================================================== SDD 256x128'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/sdd/tiny-yolov2_odgi_256_128/10-04_13-19

echo '============================================================== SDD 256x64'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/sdd/tiny-yolov2_odgi_256_64/10-05_10-26

echo '============================================================== SDD 128x64'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/sdd/tiny-yolov2_odgi_128_64/10-05_10-26