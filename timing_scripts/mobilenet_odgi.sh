#!/bin/bash
#SBATCH -N 1                            # number of nodes (usually 1)
#SBATCH -n 1                            # number of cores
#SBATCH --mail-user=aroyer@ist.ac.at    # send mail to user
#SBATCH --mail-type=FAIL,END            # if a job fails or ends
#SBATCH --mem 16G                       # memory pool for all cores
#SBATCH --time 1-00:00                  # max runtime (D-HH:MM)
#SBATCH --partition=gpu10cards          # partition (our new GPU servers)
#SBATCH --gres=gpu:0                    # how many GPUs to reserve
#SBATCH --job-name=time_mobilenet_odgi
#SBATCH -o ./log/output_logs/time_mobilenet_odgi.%j.out     # logfile for stdout
#SBATCH -e ./log/error_logs/time_mobilenet_odgi.%j.err      # logfile for stderr

echo '============================================================== ODGI 1.0-0.5 512 x 256'
python3 time_model.py ./log/sdd/mobilenet_odgi_512_256/02-05_13-16/ --mobilenet 1.0 --device 'cpu' --verbose 1

echo '============================================================== ODGI 0.5-0.5 512 x 256'
python3 time_model.py ./log/sdd/mobilenet_odgi_512_256/02-05_14-49/ --mobilenet 0.5 --device 'cpu' --verbose 1