#!/bin/bash
#SBATCH -N 1                            # number of nodes (usually 1)
#SBATCH -n 1                            # number of cores
#SBATCH --mail-user=aroyer@ist.ac.at    # send mail to user
#SBATCH --mail-type=FAIL,END            # if a job fails or ends
#SBATCH --mem 16G                       # memory pool for all cores
#SBATCH --time 1-00:00                  # max runtime (D-HH:MM)
#SBATCH --partition=gpu10cards          # partition (our new GPU servers)
#SBATCH --gres=gpu:0                    # how many GPUs to reserve
#SBATCH --job-name=time_mobilenet_standard
#SBATCH -o ./log/output_logs/time_mobilenet_standard.%j.out     # logfile for stdout
#SBATCH -e ./log/error_logs/time_mobilenet_standard.%j.err      # logfile for stderr

echo '============================================================== STANDARD 1024'
python3 time_model.py ./log/sdd/mobilenet_standard_1024/02-04_10-45/ --device 'cpu' --verbose 1

echo '============================================================== STANDARD 512'
python3 time_model.py ./log/sdd/mobilenet_standard_512/02-04_10-24/ --device 'cpu' --verbose 1