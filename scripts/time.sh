#!/bin/bash
#SBATCH -N 1                            # number of nodes (usually 1)
#SBATCH -n 1                            # number of cores
#SBATCH --mail-user=aroyer@ist.ac.at    # send mail to user
#SBATCH --mail-type=FAIL,END            # if a job fails or ends
#SBATCH --mem 16G                       # memory pool for all cores
#SBATCH --time 1-00:00                  # max runtime (D-HH:MM)
#SBATCH --partition=gpu10cards          # partition (our new GPU servers)
#SBATCH --gres=gpu:0                    # how many GPUs to reserve
#SBATCH --job-name=time_odgi
#SBATCH -o ./log/output_logs/time_odgi.%j.out     # logfile for stdout
#SBATCH -e ./log/error_logs/time_odgi.%j.err      # logfile for stderr

module load cuda/9.0
module load cudnn
module load tensorflow/python3/1.10.0
cd ${HOME}/Jupyter/ODGI                   # working directory


echo '============================================================== Standard'
python3 time_model.py 'log/vedai_fold01/tiny-yolov2_standard_128/09-26_18-24/' --device 'cpu' --verbose 1
python3 time_model.py 'log/vedai_fold01/tiny-yolov2_standard_256/09-26_12-46/' --device 'cpu' --verbose 1
python3 time_model.py 'log/vedai_fold01/tiny-yolov2_standard_512/09-24_10-16/' --device 'cpu' --verbose 1
python3 time_model.py 'log/vedai_fold01/tiny-yolov2_standard_1024/09-25_13-58/' --device 'cpu' --verbose 1

echo '============================================================== ODGI'
python3 time_model.py 'log/vedai_fold01/tiny-yolov2_odgi_128_64/09-27_10-44/' --device 'cpu' --verbose 1
python3 time_model.py 'log/vedai_fold01/tiny-yolov2_odgi_256_64/09-24_10-17/' --device 'cpu' --verbose 1
python3 time_model.py 'log/vedai_fold01/tiny-yolov2_odgi_256_128/09-25_14-48/' --device 'cpu' --verbose 1
python3 time_model.py 'log/vedai_fold01/tiny-yolov2_odgi_512_64/09-27_16-00/' --device 'cpu' --verbose 1
python3 time_model.py 'log/vedai_fold01/tiny-yolov2_odgi_512_256/09-27_20-25/' --device 'cpu' --verbose 1
exit 0
EOT