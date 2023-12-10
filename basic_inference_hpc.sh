#!/bin/sh
#BSUB -J dlbioproject
#BSUB -o  /zhome/fc/b/143004/DEEP_BIOMED/outfiles/dlbioproject_%J.out
#BSUB -e  /zhome/fc/b/143004/DEEP_BIOMED/errorfiles/dlbioproject_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1" 
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]" 
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
# end of BSUB options

module load python3/3.10.8
python3 -m venv DLBio_finalProjectv
source DLBio_finalProjectv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r /zhome/fc/b/143004/DEEP_BIOMED/requirements2.txt

cd /zhome/fc/b/143004/DEEP_BIOMED/covid_data.nosync/full_data
dvc pull
cd /zhome/fc/b/143004/DEEP_BIOMED/trained_models
dvc pull 

cd /zhome/fc/b/143004/DEEP_BIOMED/

python -u /zhome/fc/b/143004/DEEP_BIOMED/inference.py 
