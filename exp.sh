#!/bin/bash



srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4 -K 64 128 256 -L 2 -P 8 -H 100 --epochs 70 -M resnet --lr 0.005
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4 -K 64 128 256 -L 4 -P 8 -H 100 --epochs 70 -M resnet --lr 0.005
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4 -K 64 128 256 -L 8 -P 8 -H 100 --epochs 70 -M resnet --lr 0.005


