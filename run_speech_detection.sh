#!/bin/bash

#SBATCH -c 16
#SBATCH --mem=32GB
#SBATCH -p cpu-preempt
#SBATCH --time 05:00:00
#SBATCH -o run_%j.out

module load conda/latest
module load uri/main all/FFmpeg/4.2.1-GCCcore-8.3.0

conda activate ytb

python kmeans_speech.py
