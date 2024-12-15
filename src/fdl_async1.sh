#!/bin/bash

#SBATCH --job-name=my_fdl_async_job
#SBATCH --time=0-40:30:00
#SBATCH --gres=gpu:nvidia-a100:1
#SBATCH --mem-per-cpu=256G
#SBATCH --constraint a100-vram-80G
#SBATCH --output="/data/bio-eng-llm/fed_learning_wind_proj/logs/%x-%j.log"

# Change directory within the script
source /data/bio-eng-llm/virtual_envs/dnn_env/bin/activate

# Execute the Python script
python /data/bio-eng-llm/fed_learning_wind_proj/src/training_wind_psr_lstm_fl_async.py

