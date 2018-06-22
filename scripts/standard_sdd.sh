#!/bin/bash
#SBATCH -N 1                            # number of nodes (usually 1)
#SBATCH -n 4                            # number of cores
#SBATCH --mail-user=aroyer@ist.ac.at    # send mail to user
#SBATCH --mail-type=FAIL                # if a job fails
#SBATCH --mem 16G                       # memory pool for all cores
#SBATCH --time 1-00:00                  # max runtime (D-HH:MM)
#SBATCH --partition=gpu10cards          # partition (our new GPU servers)
#SBATCH --gres=gpu:2                    # how many GPUs to reserve
#SBATCH --constraint=GTX1080Ti          # GPU type (unnecessary here)
#SBATCH -o ./log/output_logs/standard-sdd.%N.%j.out     # logfile for stdout
#SBATCH -e ./log/error_logs/standard-sdd.%N.%j.err # logfile for stderr

SIZE=1024                              
DATA='stanford'                           
NUM_EPOCHS=120
NUM_GPUS=2
DISPLAY_LOSS_EVERY_N_STEPS=500
BATCH_SIZE=12
LEARNING_RATE=1e-3

module load tensorflow/python3/1.4.0      # enable tensorflow
cd ${HOME}/Jupyter/ODGI                   # working directory

python3 -u train_standard.py $DATA --size $SIZE --num_epochs=$NUM_EPOCHS --num_gpus=$NUM_GPUS --display_loss_very_n_steps=$DISPLAY_LOSS_EVERY_N_STEPS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE