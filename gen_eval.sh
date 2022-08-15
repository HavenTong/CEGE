#!/bin/bash

# CPT-base(+M2E)
python eval.py --model_name d1_cpt_pretrain_noleak_128_accum_1_10_128_10_best --top_k 1 --num_beams 5 --batch_size 1024 --file data/d1_gen_test.txt
python eval.py --model_name d1_cpt_pretrain_noleak_128_accum_1_10_128_10_best --top_k 3 --num_beams 5 --batch_size 1024 --file data/d1_gen_test.txt
python eval.py --model_name d1_cpt_pretrain_noleak_128_accum_1_10_128_10_best --top_k 12 --num_beams 32 --batch_size 256 --file data/d1_gen_test.txt

# Generate candidates for training the evaluation model
#python eval.py --model_name d1_cpt_pretrain_noleak_128_accum_1_10_128_10_best --top_k 12 --num_beams 32 --batch_size 256 --file data/d1_gen_train.txt
#python eval.py --model_name d1_cpt_pretrain_noleak_128_accum_1_10_128_10_best --top_k 12 --num_beams 32 --batch_size 256 --file data/d1_gen_val.txt

# CPT-base
#python eval.py --model_name d1_cpt-base_128_10_best --top_k 1 --num_beams 5 --batch_size 1024 --file data/d1_gen_test.txt
#python eval.py --model_name d1_cpt-base_128_10_best --top_k 3 --num_beams 5 --batch_size 1024 --file data/d1_gen_test.txt
#python eval.py --model_name d1_cpt-base_128_10_best --top_k 12 --num_beams 32 --batch_size 256 --file data/d1_gen_test.txt
