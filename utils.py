import torch
import numpy as np
import random
from collections import defaultdict
import unicodedata
import re


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_head_data(*lists, num=10000):
    return [data[:num] for data in lists]


# 对英文做归一化 "Crème Brulée" -> "Creme Brulee"
def token_normalize(token):
    return re.sub('[\u0300-\u036F]', '', unicodedata.normalize('NFKD', token))


if __name__ == '__main__':
    a = [0] * 100000
    b = [1] * 100000
    a, b = get_head_data(a, b)
    print(a)
    print(b)
    print(len(a))
    print(len(b))