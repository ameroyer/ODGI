#!/bin/bash
CUDA_DEVICE=$1


echo '============================================================== SDD 512x256'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/sdd/yolov2_odgi_512_256/10-05_14-39

echo '============================================================== SDD 256x128'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/sdd/yolov2_odgi_256_128/10-06_23-45

echo '============================================================== SDD 128x64'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py ./log/sdd/yolov2_odgi_128_64/10-06_23-46