#!/bin/bash
MODEL=$1
DEVICE=$2
if [ "$DEVICE" == "cpu" ]
    then
    NUM_GPUS=0
elif [ "$DEVICE" == "gpu" ]
    then
    NUM_GPUS=1
else
    echo "Unkown device option \"${DEVICE}\""
    exit 1
fi
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1                            # number of nodes (usually 1)
#SBATCH -n 1                            # number of cores
#SBATCH --mail-user=aroyer@ist.ac.at    # send mail to user
#SBATCH --mail-type=FAIL,END            # if a job fails or ends
#SBATCH --mem 2G                        # memory pool for all cores
#SBATCH --time 1-00:00                  # max runtime (D-HH:MM)
#SBATCH --partition=gpu10cards          # partition (our new GPU servers)
#SBATCH --gres=gpu:$NUM_GPUS            # how many GPUs to reserve
#SBATCH --constraint=GTX1080Ti          # GPU type (unnecessary here)
#SBATCH --job-name=time_${MODEL}_${DEVICE}
#SBATCH -o ./run_logs/output_logs/time_${MODEL}_${DEVICE}.%j.out     # logfile for stdout
#SBATCH -e ./run_logs/error_logs/time_${MODEL}_${DEVICE}.%j.err      # logfile for stderr

module load cuda/9.0
module load cudnn
module load tensorflow/python3/1.12.0
cd ${HOME}/Jupyter/ODGI                   # working directory

./slurm_scripts/timing_scripts/${MODEL}_timings.sh $DEVICE
EOT