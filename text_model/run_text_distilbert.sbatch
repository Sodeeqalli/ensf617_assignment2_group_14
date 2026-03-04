#!/bin/bash
#SBATCH --job-name=txt_distilbert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=text_output_model/train_%j.out

cd /home/sodeeq.alli/ensf617_a2_sodeeq
source venv/bin/activate

mkdir -p text_output_model
python -m text_model.train 2>&1 | tee text_output_model/train_log.txt