import torch
import torch.nn as nn
from preprocess import load_dataset
from config import PretrainConfig
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
import wandb


def set_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='cpt-base')
    parser.add_argument('--wandb', action="store_true")
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
    config = PretrainConfig(**vars(args))
    if config.wandb:
        wandb.init(project=f"{config.model_type}_abbr_pretrain", name=config.logging_file_name, config=vars(config))

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
    }

    remove_cols = dataset['train'].column_names

    train_set = dataset['train'].map(
        function=preprocess_function,
        batched=True,
        remove_columns=remove_cols,
        load_from_cache_file=True,
        cache_file_name=config.train_set_cache_path
    )

    val_set = dataset['val'].map(
        function=preprocess_function,
        batched=True,
        remove_columns=remove_cols,
        load_from_cache_file=True,
        cache_file_name=config.val_set_cache_path
    )

    print(train_set)
    print(val_set)

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
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=0.01,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_accumulation_steps=config.eval_accumulation_steps,
        load_best_model_at_end=True,
        logging_dir=config.tensorboard_dir,
        report_to='wandb' if config.wandb else None,
        run_name=config.logging_file_name if config.wandb else None
    )

    pprint(training_args)
    logger.info(training_args)

    trainer = Seq2SeqTrainer(
        tokenizer=config.tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
    )

    pprint(trainer)

    train_result = trainer.train()
    trainer.save_model(config.best_model_path)
    pprint(train_result)
