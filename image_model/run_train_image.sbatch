#!/bin/bash
#SBATCH --job-name=img_ft_mod
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=image_output_model/train_modular_%j.out

cd /home/sodeeq.alli/ensf617_a2_sodeeq
source venv/bin/activate

mkdir -p image_output_model
python -m image_model.train 2>&1 | tee image_output_model/train_log.txt