#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0    
#SBATCH --mail-user=<vbertalan@gmail.com>
#SBATCH --mail-type=ALL

cd ~/$projects/teste
module purge
module load python/3.8.10
source ~/.venv/bin/activate

python transformer-trainer.py