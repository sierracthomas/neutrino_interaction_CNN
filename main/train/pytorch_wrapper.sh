#!/bin/bash

eval "$(/home/$(whoami)/miniconda3/bin/conda shell.bash hook)"

conda activate pytorch

python3 -u /home/$(whoami)/neutrino_interaction_CNN/main/train/train_nn.py

