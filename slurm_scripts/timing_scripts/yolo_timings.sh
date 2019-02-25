#!/bin/bash
DEVICE=$1

echo '============================================================== YOLO'
echo '==== 1024x1024'
python3 time.py run_logs/sdd/yolo_v2_standard_1024/02-21_15-09 --device $DEVICE --verbose 0
echo '==== 512x512'
python3 time.py run_logs/sdd/yolo_v2_standard_1024/02-21_15-09 --image_size 512 --device $DEVICE --verbose 0
echo '==== 256x256'
python3 time.py run_logs/sdd/yolo_v2_standard_1024/02-21_15-09 --image_size 256 --device $DEVICE --verbose 0
echo '==== 128x128'
python3 time.py run_logs/sdd/yolo_v2_standard_1024/02-21_15-09 --image_size 128 --device $DEVICE --verbose 0

echo
echo
echo '============================================================== tiny-YOLO'
echo '==== 1024x1024'
python3 time.py run_logs/sdd/tiny_yolo_v2_standard_1024/02-21_15-09 --device $DEVICE --verbose 0
echo '==== 512x512'
python3 time.py run_logs/sdd/tiny_yolo_v2_standard_1024/02-21_15-09 --image_size 512 --device $DEVICE --verbose 0
echo '==== 256x256'
python3 time.py run_logs/sdd/tiny_yolo_v2_standard_1024/02-21_15-09 --image_size 256 --device $DEVICE --verbose 0
echo '==== 128x128'
python3 time.py run_logs/sdd/tiny_yolo_v2_standard_1024/02-21_15-09 --image_size 128 --device $DEVICE --verbose 0

echo
echo
echo '============================================================== ODGI yolo-tiny'
echo '==== 512x256'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/yolo_v2_odgi_512_256/02-21_15-09 --test_num_crops $i --device $DEVICE --verbose 0 
done
echo '==== 256x128'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/yolo_v2_odgi_512_256/02-21_15-09 --image_size 256 --stage2_image_size 128 --test_num_crops $i --device $DEVICE --verbose 0 
done
echo '==== 128x64'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/yolo_v2_odgi_512_256/02-21_15-09 --image_size 128 --stage2_image_size 64 --test_num_crops $i --device $DEVICE --verbose 0 
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
echo '==== 512x128'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/tiny_yolo_v2_odgi_512_256/02-21_15-10 --image_size 512 --stage2_image_size 128 --test_num_crops $i --device $DEVICE --verbose 0 
done
echo '==== 256x128'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/tiny_yolo_v2_odgi_512_256/02-21_15-10 --image_size 256 --stage2_image_size 128 --test_num_crops $i --device $DEVICE --verbose 0 
done
echo '==== 256x64'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/tiny_yolo_v2_odgi_512_256/02-21_15-10 --image_size 256 --stage2_image_size 64 --test_num_crops $i --device $DEVICE --verbose 0 
done
echo '==== 128x64'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/tiny_yolo_v2_odgi_512_256/02-21_15-10 --image_size 128 --stage2_image_size 64 --test_num_crops $i --device $DEVICE --verbose 0 
done