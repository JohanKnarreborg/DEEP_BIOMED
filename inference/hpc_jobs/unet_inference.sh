#!/bin/sh
#BSUB -J dlbioproject
#BSUB -o  /zhome/f9/2/183623/DEEP_BIOMED/outfiles/dlbioproject_%J.out
#BSUB -e  /zhome/f9/2/183623/DEEP_BIOMED/errorfiles/dlbioproject_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1" 
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
# end of BSUB options

module load python3/3.8.17
python3 -m venv /zhome/f9/2/183623/DEEP_BIOMED/DLBio_inferencev
source /zhome/f9/2/183623/DEEP_BIOMED/DLBio_inferencev/bin/activate
python3 -m pip install -r /zhome/f9/2/183623/DEEP_BIOMED/inference/requirements2.txt

cd /zhome/f9/2/183623/DEEP_BIOMED/covid_data.nosync/full_data
dvc pull

cd /zhome/f9/2/183623/DEEP_BIOMED/

python -u /zhome/f9/2/183623/DEEP_BIOMED/inference/inference.py --model_type unet --data_path /zhome/f9/2/183623/DEEP_BIOMED/covid_data.nosync/full_data/train/data_0.npy --wandb_runtime 2023-12-09_23-51
