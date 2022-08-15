#!/bin/bash

# Train from CPT-base(+M2E)
python run.py --batch_size 128 --num_epochs 10 --model_name cpt_pretrain_noleak_128_accum_1_10

# Train from CPT-base
#python run.py --batch_size 128 --num_epochs 10 --model_name cpt-base

sh gen_eval.sh
