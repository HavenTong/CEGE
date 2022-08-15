from tqdm import tqdm
import torch
import torch.nn as nn
import thwpy


def train(data_loader, model, loss, optimizer, device, config, scheduler=None):
    model.train()
    final_loss = 0.0
    steps = 0
    bar = tqdm(data_loader, desc='train')
    for data in bar:
        for k, v in data.items():
            data[k] = v.to(device)
        label = data.pop('labels')

        outputs = model(**data)
        ls = loss(outputs, label)

        ls = ls / config.gradient_accumulation_steps
        ls.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_norm)

        if ((steps + 1) % config.gradient_accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        final_loss += ls.item()
        steps += 1
        bar.set_postfix({'avg_loss': final_loss / steps})
    return final_loss / steps


def accuracy(data_loader, model, device, topk=3):
    model.eval()
    predicts = []
    hit3_predicts = []
    labels = []
    bar = tqdm(data_loader, desc='accuracy')
    with torch.no_grad():
        for data in bar:
            for k, v in data.items():
                data[k] = v.to(device)

            label = data.pop('labels')
            outputs = model(**data)

            logit, preds = torch.topk(outputs, topk, dim=-1)
            hit3_predicts.extend(preds.tolist())

            labels.extend(label.tolist())

    hit_1 = sum(j == i[0] for i, j in zip(hit3_predicts, labels)) / len(labels)
    hit_3 = sum(j in i for i, j in zip(hit3_predicts, labels)) / len(labels)

    return hit_1, hit_3


def save_ranker_predicts(data_loader, model, device, topk=3):
    model.eval()
    predicts = []
    hit3_predicts = []
    labels = []
    bar = tqdm(data_loader, desc='accuracy')
    with torch.no_grad():
        for data in bar:
            for k, v in data.items():
                data[k] = v.to(device)

            label = data.pop('labels')
            outputs = model(**data)

            logit, preds = torch.topk(outputs, topk, dim=-1)
            hit3_predicts.extend(preds.tolist())

            labels.extend(label.tolist())

    hit_1 = sum(j == i[0] for i, j in zip(hit3_predicts, labels)) / len(labels)
    hit_3 = sum(j in i for i, j in zip(hit3_predicts, labels)) / len(labels)

    return hit_1, hit_3, hit3_predicts