#!/bin/bash
#SBATCH -o ./dummy_logs/slurm-%j.out 
NETWORK=$1
SIZE=$2     
NUM_EPOCHS=100
LEARNING_RATE=1e-3

### Set network-dependent parameters
if [ "$NETWORK" == "tiny_yolo_v2" ]
    then
    NUM_GPUS=2
    BATCH_SIZE=12   
elif [ "$NETWORK" == "yolo_v2" ]
    then
    NUM_GPUS=4
    BATCH_SIZE=4   
elif [ "$NETWORK" == "mobilenet_100" ] || [ "$NETWORK" == "mobilenet_50" ] || [ "$NETWORK" == "mobilenet_35" ]
    then
    NUM_GPUS=4
    BATCH_SIZE=8
    LEARNING_RATE=2e-4
else
    echo "Unkown network option \"${NETWORK}\""
    exit 1
fi

sbatch <<EOT
#!/bin/bash
#SBATCH -N 1                            # number of nodes (usually 1)
#SBATCH -n 2                            # number of cores
#SBATCH --mail-user=aroyer@ist.ac.at    # send mail to user
#SBATCH --mail-type=FAIL,END            # if a job fails or ends
#SBATCH --mem 32G                       # memory pool for all cores
#SBATCH --time 5-00:00                  # max runtime (D-HH:MM)
#SBATCH --partition=gpu10cards          # partition (our new GPU servers)
#SBATCH --gres=gpu:$NUM_GPUS            # how many GPUs to reserve
#SBATCH --constraint=GTX1080Ti          # GPU type (unnecessary here)
#SBATCH --job-name=sdd_${NETWORK}_${SIZE}
#SBATCH -o ./run_logs/output_logs/sdd_${NETWORK}_${SIZE}.%j.out     # logfile for stdout
#SBATCH -e ./run_logs/error_logs/sdd_${NETWORK}_${SIZE}.%j.err      # logfile for stderr

module load cuda/9.0
module load cudnn
module load tensorflow/python3/1.12.0

cd ${HOME}/Jupyter/ODGI
python3 -u train_standard.py 'sdd' --network $NETWORK --image_size $SIZE --num_epochs $NUM_EPOCHS --num_gpus $NUM_GPUS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE
EOT