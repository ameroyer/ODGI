#!/bin/bash
#SBATCH -N 1                            # number of nodes (usually 1)
#SBATCH -n 4                            # number of cores
#SBATCH --mail-user=aroyer@ist.ac.at    # send mail to user
#SBATCH --mail-type=FAIL,END            # if a job fails or ends
#SBATCH --mem 32G                       # memory pool for all cores
#SBATCH --time 1-00:00                  # max runtime (D-HH:MM)
#SBATCH --partition=gpu10cards          # partition (our new GPU servers)
#SBATCH --gres=gpu:2                    # how many GPUs to reserve
#SBATCH --constraint=GTX1080Ti          # GPU type (unnecessary here)
#SBATCH -o ./log/output_logs/odgi-vedai.%N.%j.out     # logfile for stdout
#SBATCH -e ./log/error_logs/odgi-vedai.%N.%j.err # logfile for stderr

SIZE=512                              
DATA='vedai'                           
NUM_EPOCHS=1000
NUM_GPUS=2
DISPLAY_LOSS_EVERY_N_STEPS=250
BATCH_SIZE=16
FULL_IMAGE_SIZE=0
LEARNING_RATE=1e-3
STAGE2_MOMENTUM=0.65

module load tensorflow/python3/1.4.0      # enable tensorflow
cd ${HOME}/Jupyter/ODGI                   # working directory

python3 -u train_odgi.py $DATA --size $SIZE --num_epochs=$NUM_EPOCHS --num_gpus=$NUM_GPUS --display_loss_very_n_steps=$DISPLAY_LOSS_EVERY_N_STEPS --batch_size=$BATCH_SIZE --full_image_size=$FULL_IMAGE_SIZE --learning_rate=$LEARNING_RATE --stage2_momentum=$STAGE2_MOMENTUM
