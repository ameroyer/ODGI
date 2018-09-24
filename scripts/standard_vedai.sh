#!/bin/bash
SIZE=512        
NETWORK='tiny-yolov2'

NUM_GPUS=2
BATCH_SIZE=16       
NUM_EPOCHS=600
LEARNING_RATE=1e-3
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
#SBATCH -o ./log/output_logs/standard-vedai_$NETWORK-$SIZE-$NUM_EPOCHS.%j.out     # logfile for stdout
#SBATCH -e ./log/error_logs/standard-vedai_$NETWORK-$SIZE-$NUM_EPOCHS.%j.err      # logfile for stderr

module load cuda/9.0
module load cudnn
module load tensorflow/python3/1.10.0
cd ${HOME}/Jupyter/ODGI                   # working directory

echo '============================================================== FOLD 01'
python3 -u train_standard.py 'vedai_fold01' --network=$NETWORK --size=$SIZE --num_epochs=$NUM_EPOCHS --num_gpus=$NUM_GPUS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE   
echo
echo

echo '============================================================== FOLD 02'
python3 -u train_standard.py 'vedai_fold02' --network=$NETWORK --size=$SIZE --num_epochs=$NUM_EPOCHS --num_gpus=$NUM_GPUS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE 
echo
echo

echo '============================================================== FOLD 03' 
python3 -u train_standard.py 'vedai_fold03' --network=$NETWORK --size=$SIZE --num_epochs=$NUM_EPOCHS --num_gpus=$NUM_GPUS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE  
echo
echo

echo '============================================================== FOLD 04'
python3 -u train_standard.py 'vedai_fold04' --network=$NETWORK --size=$SIZE --num_epochs=$NUM_EPOCHS --num_gpus=$NUM_GPUS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE   
echo
echo

echo '============================================================== FOLD 05'
python3 -u train_standard.py 'vedai_fold05' --network=$NETWORK --size=$SIZE --num_epochs=$NUM_EPOCHS --num_gpus=$NUM_GPUS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE   
echo
echo

echo '============================================================== FOLD 06'
python3 -u train_standard.py 'vedai_fold06' --network=$NETWORK --size=$SIZE --num_epochs=$NUM_EPOCHS --num_gpus=$NUM_GPUS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE   
echo
echo

echo '============================================================== FOLD 07'
python3 -u train_standard.py 'vedai_fold07' --network=$NETWORK --size=$SIZE --num_epochs=$NUM_EPOCHS --num_gpus=$NUM_GPUS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE   
echo
echo

echo '============================================================== FOLD 08'
python3 -u train_standard.py 'vedai_fold08' --network=$NETWORK --size=$SIZE --num_epochs=$NUM_EPOCHS --num_gpus=$NUM_GPUS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE 
echo
echo

echo '============================================================== FOLD 09'
python3 -u train_standard.py 'vedai_fold09' --network=$NETWORK --size=$SIZE --num_epochs=$NUM_EPOCHS --num_gpus=$NUM_GPUS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE  
echo
echo

echo '============================================================== FOLD 010'
python3 -u train_standard.py 'vedai_fold10' --network=$NETWORK --size=$SIZE --num_epochs=$NUM_EPOCHS --num_gpus=$NUM_GPUS --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE    
echo
exit 0
EOT