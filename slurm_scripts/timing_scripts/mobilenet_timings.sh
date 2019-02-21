#!/bin/bash
DEVICE=$1

echo '============================================================== Mobilenet_100'
echo '==== 1024x1024'
python3 time.py run_logs/sdd/mobilenet_100_standard_1024/02-19_13-08 --device $DEVICE --verbose 0
echo
echo '==== 512x512'
python3 time.py run_logs/sdd/mobilenet_100_standard_512/02-19_08-59 --device $DEVICE --verbose 0
echo
echo '==== 256x256'
python3 time.py run_logs/sdd/mobilenet_100_standard_256/02-19_16-44 --device $DEVICE --verbose 0

echo
echo
echo '============================================================== Mobilenet_35'
echo '==== 1024x1024'
python3 time.py run_logs/sdd/mobilenet_35_standard_1024/02-20_16-54 --device $DEVICE --verbose 0
echo
echo '==== 512x512'
python3 time.py run_logs/sdd/mobilenet_35_standard_512/02-20_16-55 --device $DEVICE --verbose 0
echo
echo '==== 256x256'
python3 time.py run_logs/sdd/mobilenet_35_standard_256/02-21_10-08 --device $DEVICE --verbose 0

echo
echo
echo '============================================================== ODGI Mobilenet_100_35'
echo '==== 512x256'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/mobilenet_100_odgi_512_256/02-19_20-17 --test_num_crops $i --device $DEVICE --verbose 0 
done
echo
echo '==== 512x128'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/mobilenet_100_odgi_512_128/02-20_10-54 --test_num_crops $i --device $DEVICE --verbose 0 
done
echo
echo '==== 512x64'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/mobilenet_100_odgi_512_64/02-21_10-11 --test_num_crops $i --device $DEVICE --verbose 0 
done
echo
echo '==== 256x128'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/mobilenet_100_odgi_256_128/02-20_10-54 --test_num_crops $i --device $DEVICE --verbose 0 
done

echo
echo
echo '============================================================== ODGI Mobilenet_35_35'
echo '==== 512x256'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/mobilenet_35_odgi_512_256/02-20_10-55 --test_num_crops $i --device $DEVICE --verbose 0 
done
echo
echo '==== 512x128'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/mobilenet_35_odgi_512_128/02-20_11-51 --test_num_crops $i --device $DEVICE --verbose 0 
done
echo
echo '==== 512x64'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/mobilenet_35_odgi_512_64/02-21_10-11 --test_num_crops $i --device $DEVICE --verbose 0 
done
echo
echo '==== 256x128'
for i in $(seq 1 10); 
do 
    echo '> Using' $i 'crops----------------------'
    python3 time.py run_logs/sdd/mobilenet_35_odgi_256_128/02-20_10-56 --test_num_crops $i --device $DEVICE --verbose 0 
done