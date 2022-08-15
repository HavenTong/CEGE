import torch
import torch.nn as nn
from preprocess import load_dataset
from config import Config
import logging
from utils import seed_everything
from argparse import ArgumentParser
from pprint import pprint
from transformers import (
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    TrainingArguments
)
from modeling_cpt import CPTForConditionalGeneration
import numpy as np
import wandb


def set_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--model_name', type=str, default='cpt-base', help='specify pre-trained models')
    parser.add_argument('--dataset', type=str, default='d1')
    parser.add_argument('--wandb', action="store_true", help="whether to use wandb")
    return parser.parse_args()


def preprocess_function(examples):
    full = examples['full']
    abbr = examples['abbr']
    inputs = config.tokenizer(full, padding=False, truncation=True)

    # Set up the tokenizer for the targets
    with config.tokenizer.as_target_tokenizer():
        labels = config.tokenizer(abbr, padding=False, truncation=True)

    inputs['labels'] = labels['input_ids']
    return inputs


if __name__ == '__main__':
    seed_everything()
    args = set_args()
    pprint(args)
    config = Config(**vars(args))
    if config.wandb:
        wandb.init(project=f"{config.dataset}_{config.model_type}_abbr_gen", name=config.logging_file_name, config=vars(config))

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=config.logging_file_name,
        filemode='a+',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    dataset = {
        'train': load_dataset(config.train_path),
        'val': load_dataset(config.val_path),
        'test': load_dataset(config.test_path)
    }

    remove_cols = dataset['train'].column_names

    train_set = dataset['train'].map(
        function=preprocess_function,
        batched=True,
        remove_columns=remove_cols,
        load_from_cache_file=False
    )

    val_set = dataset['val'].map(
        function=preprocess_function,
        batched=True,
        remove_columns=remove_cols,
        load_from_cache_file=False
    )

    test_set = dataset['test'].map(
        function=preprocess_function,
        batched=True,
        remove_columns=remove_cols,
        load_from_cache_file=False
    )

    print(train_set)
    print(val_set)
    print(test_set)
    print(train_set[:5])

    if config.model_type == 'bart':  # BART
        model = BartForConditionalGeneration.from_pretrained(config.model_name)
    else:  # CPT
        model = CPTForConditionalGeneration.from_pretrained(config.model_name)

    label_pad_token_id = -100 if config.ignore_pad_token_for_loss else config.tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=config.tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.save_path,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=0.01,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        learning_rate=config.learning_rate,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model='hit@1',
        logging_dir=config.tensorboard_dir,
        report_to='wandb' if config.wandb else None,
        run_name=config.logging_file_name if config.wandb else None
    )

    pprint(training_args)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = config.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if config.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, config.tokenizer.pad_token_id)
        decoded_labels = config.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip().replace(' ', '') for pred in decoded_preds]
        decoded_labels = [label.strip().replace(' ', '') for label in decoded_labels]

        assert len(decoded_preds) == len(decoded_labels)

        acc = sum(i == j for i, j in zip(decoded_preds, decoded_labels)) / len(decoded_labels)
        return {'hit@1': acc}


    trainer = Seq2SeqTrainer(
        tokenizer=config.tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    pprint(trainer)

    train_result = trainer.train()
    trainer.save_model(config.best_model_path)
    # pprint(train_result)

    if trainer.is_world_process_zero():
        prediction, labels, metrics = trainer.predict(test_set, metric_key_prefix='predict', num_beams=5)
        # pprint(prediction)
        # pprint(prediction.shape)
        # pprint(labels)
        # pprint(metrics)
        logger.info(prediction)
        logger.info(labels)
        logger.info(metrics)
        test_preds = config.tokenizer.batch_decode(
            prediction, skip_special_tokens=True
        )
        test_preds = [pred.strip().replace(' ', '') for pred in test_preds]
        print(len(test_preds))
        with open(config.predict_file, 'w+', encoding='utf-8') as f:
            f.write("\n".join(test_preds))