#!/bin/bash
CUDA_DEVICE=$1


echo '============================================================== SDD 512x256'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py --no_offsets ./log/sdd/tiny-yolov2_odginogroups_512_256/10-10_13-57 

echo '============================================================== SDD 256x128'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py --no_offsets ./log/sdd/tiny-yolov2_odginogroups_256_128/10-10_14-06/

echo '============================================================== SDD 128x64'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 validate_odgi.py --no_offsets ./log/sdd/tiny-yolov2_odginogroups_128_64/10-10_14-06/