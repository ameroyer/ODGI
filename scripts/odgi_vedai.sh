#!/bin/bash
SIZE=256               

NUM_GPUS=2
BATCH_SIZE=16       
NUM_EPOCHS=1000
LEARNING_RATE=1e-3
DISPLAY_LOSS_EVERY_N_STEPS=250
sbatch <<EOT
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
#SBATCH -o ./log/output_logs/odgi-vedai-$SIZE-$NUM_EPOCHS.%j.out     # logfile for stdout
#SBATCH -e ./log/error_logs/odgi-vedai-$SIZE-$NUM_EPOCHS.%j.err      # logfile for stderr

module load tensorflow/python3/1.4.0      # enable tensorflow
cd ${HOME}/Jupyter/ODGI                   # working directory

python3 -u train_odgi.py 'vedai'                            \
    --size=$SIZE                                            \
    --num_epochs=$NUM_EPOCHS                                \
    --num_gpus=$NUM_GPUS                                    \
    --display_loss_very_n_steps=$DISPLAY_LOSS_EVERY_N_STEPS \
    --batch_size=$BATCH_SIZE                                \
    --full_image_size=1024                                  \
    --learning_rate=$LEARNING_RATE                          \
    --stage2_momentum=0.65    
exit 0
EOT