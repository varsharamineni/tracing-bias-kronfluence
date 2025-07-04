#!/bin/bash
#SBATCH --job-name=kronfluence_fit_score
#SBATCH --output=fit_and_score.%j.out
#SBATCH --error=fit_and_score.%j.err
#SBATCH --time=01:00:00       # adjust as needed
#SBATCH --partition=boost_usr_prod   # update if needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1          # request GPU if needed
#SBATCH --account=euhpc_d19_099  # your account

# Load any required modules
module purge
module load python/3.10   # or your required python module
# module load cuda/xx      # if you need CUDA module

# Activate your virtual environment
source /leonardo/home/userexternal/vraminen/envs/kronfluence/bin/activate

# Go to your project directory (replace with the right path)
cd /leonardo/home/userexternal/vraminen/tracing-bias-kronfluence/examples/AnthropicHH-Bias

# Run fine-tuning
echo "Starting fine-tuning..."
python SFT_Trainer_Lora.py || { echo "fine_tune.py failed"; exit 1; }

echo "Fine-tuning complete."

echo "All done."