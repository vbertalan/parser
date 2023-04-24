#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0    
#SBATCH --mail-user=<vbertalan@gmail.com>
#SBATCH --mail-type=ALL

cd /home/vberta/projects/def-aloise/vberta/Parser/parser
source .venv/bin/activate
cd /home/vberta/projects/def-aloise/vberta/Parser-CC/parser

python transformer-trainer.py