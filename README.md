# CEGE

This repository contains the source code and datasets for the paper: **A Context-Enhanced Generate-then-Evaluate Framework for Chinese Abbreviation Prediction,** CIKM 2022.

## Enviroment Details

Some main dependencies: 

- python=3.7.11
- pytorch=1.8.2 (LTS，can be installed from [PyTorch](https://pytorch.org/get-started/locally/) website)
- transformers=4.18.0
- pandas
- datasets
- jieba
- wandb

We also provide `requirements.txt`. You can install the dependencies as follows:

```bash
conda create -n cege python=3.7
conda activate cege
pip install -r requirements.txt 
```

## Project Structure

**All the data files follow the `tsv` format, i.e., each column is separated by `\t`.**

- `data/` All the data files. 
    - `d1.txt` and `d1_gen.txt` are the whole datasets without splitting. Note that `d1.txt` is identical to data from [this repo](https://github.com/leroLiu/Sequence-to-Sequence-Learning-for-Chinese-Abbreviation-Prediction), `d1_gen.txt` is the processed one.
    - `d1_{split}.txt ` Raw datasets. The columns are `[src, label_sequence]`.
    - `d1_gen_{split}.txt` Datasets for the generation model. The columns are `[src, target]`.
    - `d1_v1_ranker_extract_all_truncate150_top12_{split}.txt` Datasets for the evaluation model. The columns are `[src, target, context, candidates, label]`. Note that the `candidates` are generated by the generation model and heuristic rules.
- `eval/`:  The predictions and results of models during evaluation.
- `config.py`:  The configuration for training and evalating the models.
- `model.py`:  Models.
- `thwpy.py`, `utils.py`:  Utilities.
- `preprocess.py`:  Data preprocessing.
- `train_eval.py`:  The functions for training and evaluating the models.
- `run*.py`:  Train the models.
- `run.py` Train the generation model.
    - `run_pretrain.py` Pre-training the generation model with Mention2Entity data.
    - `run_ranker.py` Train the evaluation model.
- `eval.py`:  Evaluate the generation model. The predictions will be stored in `eval/` and the results will be written in `eval/eval_result.txt`.

## Pre-trained Language Models

The generation model：[cpt-base](https://huggingface.co/fnlp/cpt-base)

The evaluation model：[chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base)

Download the weights and put them in `./`.

Note that our generation model is additionally pre-trained on [Mention2Entity](https://www.scidb.cn/en/detail?dataSetId=663738622406557696&code=5e05cb5d64a42fa9add9b7ae&tID=journalOne&dataSetType=journal&language=en_US) data from CN-DBpedia. We ensure there is no data leakage in the pre-training data. The weights can be downloaded [here](https://drive.google.com/drive/folders/1oow13Mhv2BAE1lr8Rgx5OiBk17lJI3_v?usp=sharing).

## How to use

### 1. Train & Evaluate the generation model

The scripts are in `train_gen.sh`:

```bash
sh train_gen.sh
```

- The paths to datasets are specified in `config.Config`. The format of dataset is `[src, target]`. The model saving path is specified in `config.Config.best_model_path`.

- `gen_eval.sh` ：
    - `--model_name`: Model for evaluating.
    - `--file`: test file in format `[src, target]`，e.g. `data/d1_gen_test.txt`.


### 2. Train & Evaluate the evaluation model

The scripts are in `train_ranker.sh`:

```bash
sh train_ranker.sh
```

- The paths to datasets are specified in `config.RankerConfig`. Note that the path can be changed under different settings. The format of dataset is: `[src, target, context, candidates, label]`, e.g. `data/d1_v1_ranker_extract_all_truncate150_top12_test.txt`
- `config.RankerConfig.save_path` specifies the path to save the model. `config.RankerConfig.logging_file_name` specifies the path to logs.


