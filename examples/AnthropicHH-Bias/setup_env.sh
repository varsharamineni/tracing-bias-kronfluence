#!/bin/bash
echo "🔧 Starting setup_env.sh..."

# Load compatible Python module (not cineca-ai)
module purge
module load profile/base
module load python/3.10.8--gcc--8.5.0

# Define and create venv
ENV_DIR=~/envs/kronfluence

if [ ! -d "$ENV_DIR" ]; then
    echo "📦 Creating virtual environment at $ENV_DIR"
    python -m venv "$ENV_DIR"
else
    echo "✅ Virtual environment already exists at $ENV_DIR"
fi

# Activate venv
source "$ENV_DIR/bin/activate"

# Install clean packages inside your venv
pip install --upgrade pip
pip install -r requirements.txt
pip install kronfluence==1.0.1

echo "✅ Environment setup complete."