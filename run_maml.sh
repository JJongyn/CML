#!/bin/bash

# Base Model -> MAML

echo "[Start meta-training MAML ...]"
python ./train_maml.py --folder=~/data \
                 --dataset=miniimagenet \
                 --model=4-conv \
                 --hidden-size=64 \
                 --device=cuda:0 \
                 --num-ways=5 \
                 --num-shots=5 \
                 --extractor-step-size=0.5 \
                 --classifier-step-size=0.5 \
                 --meta-lr=1e-3 \
                 --download \
                 --batch-iter=300 \
                 --output-folder=./result \
                 --save-name=MAML

echo "[Start meta-testing MAML ...]"
python ./test_maml.py --folder=~/data \
                 --dataset=miniimagenet \
                 --model=4-conv \
                 --device=cuda:0 \
                 --num-ways=5 \
                 --num-shots=5 \
                 --extractor-step-size=0.5 \
                 --classifier-step-size=0.5 \
                 --meta-lr=1e-3 \
                 --download \
                 --output-folder=./result \
                 --save-name=MAML

echo "finished"
