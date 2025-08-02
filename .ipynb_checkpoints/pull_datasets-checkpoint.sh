#!/bin/bash

# Script to pull bias evaluation datasets
echo "Pulling bias evaluation datasets..."

# Create datasets directory
mkdir -p datasets

# CrowS-Pairs
echo "Pulling CrowS-Pairs..."
if [ ! -d "datasets/crows-pairs" ]; then
    git clone https://github.com/nyu-mll/crows-pairs.git datasets/crows-pairs
else
    echo "CrowS-Pairs already exists, skipping..."
fi

# BiasBench (replacing StereoSet)
echo "Pulling BiasBench..."
if [ ! -d "datasets/bias-bench" ]; then
    git clone https://github.com/McGill-NLP/bias-bench.git datasets/bias-bench
else
    echo "BiasBench already exists, skipping..."
fi

# Winobias
echo "Pulling Winobias..."
if [ ! -d "datasets/winobias" ]; then
    git clone https://github.com/rudinger/winobias.git datasets/winobias
else
    echo "Winobias already exists, skipping..."
fi

# Winogender
echo "Pulling Winogender..."
if [ ! -d "datasets/winogender" ]; then
    git clone https://github.com/rudinger/winogender.git datasets/winogender
else
    echo "Winogender already exists, skipping..."
fi

# Bias in Bios
echo "Pulling Bias in Bios..."
if [ ! -d "datasets/biosbias" ]; then
    git clone https://github.com/microsoft/biosbias.git datasets/biosbias
else
    echo "Bias in Bios already exists, skipping..."
fi

# BOLD
echo "Pulling BOLD..."
if [ ! -d "datasets/bold" ]; then
    git clone https://github.com/amazon-research/bold.git datasets/bold
else
    echo "BOLD already exists, skipping..."
fi

# BBQ
echo "Pulling BBQ..."
if [ ! -d "datasets/bbq" ]; then
    git clone https://github.com/nyu-mll/bbq.git datasets/bbq
else
    echo "BBQ already exists, skipping..."
fi

echo "All datasets pulled successfully!"
echo "Datasets are located in the 'datasets' directory." 