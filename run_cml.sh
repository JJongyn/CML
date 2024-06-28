#!/bin/bash

# Base Model -> MAML

echo "[Start meta-training CML ...]"
python ./train_cml.py --folder=~/data \
                 --dataset=miniimagenet \
                 --model=4-conv_cml \
                 --hidden-size=64 \
                 --device=cuda:0 \
                 --num-ways=5 \
                 --num-shots=5 \
                 --extractor-step-size=0.5 \
                 --classifier-step-size=0.5 \
                 --meta-lr=1e-3 \
                 --download \
                 --batch-iter=300 \
                 --loss-scaling=1 \
                 --output-folder=./result \
                 --save-name=CML

echo "[Start meta-testing CML ...]"
python ./test_cml.py --folder=~/data \
                 --dataset=miniimagenet \
                 --model=4-conv_cml \
                 --device=cuda:0 \
                 --num-ways=5 \
                 --num-shots=5 \
                 --extractor-step-size=0.5 \
                 --classifier-step-size=0.5 \
                 --meta-lr=1e-3 \
                 --download \
                 --output-folder=./result \
                 --save-name=CML \
                 --use-colearner

echo "finished"
