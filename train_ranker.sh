#!/bin/bash

python run_ranker.py --version 1 --batch_size 4 --num_epochs 10 --gradient_accumulation_steps 512 --lr 4e-5 --top_k 12