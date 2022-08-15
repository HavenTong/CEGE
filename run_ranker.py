import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import RankerConfig
from preprocess import read_ranker_data
from model import Ranker
from dataset import RankerDataset
from train_eval import train, accuracy
from utils import seed_everything
import argparse
from pprint import pprint
import logging
from transformers import get_linear_schedule_with_warmup, AdamW
import os
import wandb


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='d1')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--mode', type=str, default='add_b',
                        help='how to fill the candidates into context, see read_ranker_data()')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=512)
    parser.add_argument('--ablation', type=str, default='all', help='ablation on the heuristic rules')
    parser.add_argument('--top_k', type=int, default=12, help='number of candidates')
    parser.add_argument('--truncation', type=int, default=150, help='truncation length of context')
    parser.add_argument('--no_extract', action='store_true', help='whether to extract sentences in context')
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
    return parser.parse_args()


if __name__ == '__main__':
    seed_everything()
    args = set_args()
    pprint(args)
    config = RankerConfig(**vars(args))
    if config.wandb:
        wandb.init(project=f'{config.dataset}_cpt_abbr_ranker', name=config.logging_file_name, config=vars(config))

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=config.logging_file_name,
        filemode='a+',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    train_references, train_contexts, train_labels = read_ranker_data(config.train_path, mode=config.mode)
    val_references, val_contexts, val_labels = read_ranker_data(config.val_path, mode=config.mode)
    test_references, test_contexts, test_labels = read_ranker_data(config.test_path, mode=config.mode)

    train_set = RankerDataset(train_references, train_contexts, train_labels, config)
    val_set = RankerDataset(val_references, val_contexts, val_labels, config)
    test_set = RankerDataset(test_references, test_contexts, test_labels, config)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, collate_fn=val_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=test_set.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Ranker(config).to(device)
    loss = nn.CrossEntropyLoss(ignore_index=-1)

    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_params = [
        {
            'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_params, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=config.num_epochs * len(train_loader))

    logger.info("START")
    best_metric = 0.0
    best_model_name = ''
    if config.wandb:
        wandb.watch(model)
    for epoch in range(config.num_epochs):
        train_loss = train(train_loader, model, loss, optimizer, device, config, scheduler)
        logger.info(f"Epoch: [{epoch + 1} / {config.num_epochs}] | Train Loss: {train_loss}")
        val_hit1, val_hit3 = accuracy(val_loader, model, device)
        logger.info(f"Epoch: [{epoch + 1} / {config.num_epochs}] | Val Hit@1: {val_hit1} | Val Hit@3: {val_hit3}")
        if config.wandb:
            wandb.log({'loss': train_loss})
            wandb.log({'Hit@1(val)': val_hit1, 'Hit@3(val)': val_hit3, 'avg(val)': (val_hit1 + val_hit3) / 2})
        if (val_hit1 + val_hit3) > best_metric:
            best_metric = val_hit1 + val_hit3
            if os.path.exists(best_model_name):
                os.remove(best_model_name)
            best_model_name = f"{config.save_path.split('.')[0]}_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), best_model_name)
            logger.info(f"Model saved in {best_model_name} @Epoch {epoch + 1}")

    model.load_state_dict(torch.load(best_model_name))
    hit_1, hit_3 = accuracy(test_loader, model, device)
    if config.wandb:
        wandb.log({"Hit@1(test)": hit_1, "Hit@3(test)": hit_3})
    logger.info(f"Best Model '{best_model_name}' | Test Hit@1: {hit_1} | Test Hit@3: {hit_3}")
