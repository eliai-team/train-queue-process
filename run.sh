#!/bin/bash

git clone https://github.com/eliai-team/train-queue-process train-script

cd train-script
pip install -e .


python3 -u train_queue.py