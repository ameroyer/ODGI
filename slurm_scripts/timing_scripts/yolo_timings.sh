#!/bin/bash
DEVICE=$1

echo '============================================================== YOLO'
echo '==== 1024x1024'
python3 time.py run_logs/sdd/yolo_v2_standard_1024/02-21_15-09 --device $DEVICE --verbose 0

echo
echo
echo '============================================================== tiny-YOLO'
echo '==== 1024x1024'
python3 time.py run_logs/sdd/tiny_yolo_v2_standard_1024/02-21_15-09 --device $DEVICE --verbose 0

echo
echo
echo '============================================================== ODGI yolo-tiny'
echo '==== 512x256'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/yolo_v2_odgi_512_256/02-21_15-09 --test_num_crops $i --device $DEVICE --verbose 0 
done

echo
echo
echo '============================================================== ODGI teeny-tiny'
echo '==== 512x256'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/tiny_yolo_v2_odgi_512_256/02-21_15-10 --test_num_crops $i --device $DEVICE --verbose 0 
done