from datasets import Dataset
import pandas as pd
import thwpy
import random
from utils import seed_everything
import numpy as np
import re
from pprint import pprint
from itertools import combinations
import jieba
import json
from collections import Counter

seed_everything()
PLH = '[PLH]'
PREFIX = f'以下是{PLH}的介绍：'
MID_EDIT_DISTANCE = 4
UNK = '[UNK]'
GEN_DATA_NAMES = ['src', 'target']
CANDIDATE_DATA_NAMES = ['src', 'target', 'candidates']
CANDIDATE_WITH_LABEL_DATA_NAMES = ['src', 'target', 'candidates', 'label']
RANKER_DATA_NAMES = ['src', 'target', 'context', 'candidates', 'label']
RANKER_NO_CONTEXT_DATA_NAMES = ['src', 'target', 'candidates', 'label']
RAW_CONTEXT_DATA_NAMES = ['src', 'target', 'context']
CONTEXT_DATA_NAMES = ['src', 'target', 'context', 'context_with_placeholder']
ABLATIONS = ['noheu', 'noword', 'notruth', 'nochar', 'all']
header_name_dict = {
    'candidate': CANDIDATE_DATA_NAMES,
    'gen': GEN_DATA_NAMES,
    'ranker': RANKER_DATA_NAMES,
    'context': CONTEXT_DATA_NAMES,
    'raw_context': RAW_CONTEXT_DATA_NAMES
}


def read_text_pairs(file):
    data = pd.read_csv(file, sep='\t', names=['src', 'target'])
    return data['src'].tolist(), data['target'].tolist()


def read_ranker_data(file, mode='add_b'):
    # add_b: fill candidates into context
    # none: no context
    assert mode in ['add_b', 'none']

    data = pd.read_csv(file, sep='\t', names=RANKER_DATA_NAMES)
    if data['label'].isna().any():
        data = data.drop(columns=['candidates', 'label'])
        data = data.rename(columns={'context': 'candidates', 'candidates': 'label'})

    assert all(not data[key].isna().any() for key in data.keys())

    data['candidates'] = data['candidates'].map(lambda x: x.split(';'))
    if mode != 'none':
        references = [context.replace(PLH, full) for context, full in
                      zip(data['context'], data['src'])]
    else:
        references = data['src'].tolist()
    if mode == 'add_b':
        contexts = [[context.replace(PLH, f"{full}（简称“{candidate}”）") for candidate in candidates] for
                    candidates, full, abbr, context in
                    zip(data['candidates'], data['src'], data['target'], data['context'])]
    else:
        contexts = [[f"{full}（简称“{candidate}”）" for candidate in candidates]
                    for full, candidates in zip(data['src'], data['candidates'])]

    assert len(references) == len(contexts) and len(contexts) == len(data['label'])

    return references, contexts, data['label'].tolist()


def rfind(s, find_chars):
    """从s的尾部开始，找到第一个满足 在find_chars中的字符 的位置"""
    pos = len(s) - 1
    while pos >= 0 and s[pos] not in find_chars:
        pos -= 1
    return pos


def truncate_context_len(file, max_len, truncate_len, save=False):
    """
    对带有PLH的context进行truncate
    :param save:
    :param file: 带有PLH的context文件
    :param max_len: 对于大于max_len的context需要进行truncate，truncate的方式就是直接截断
    :param truncate_len: 截断的长度，即选取context[:truncate_len]
    :return:
    """
    file_split = thwpy.check_split(file)
    data = pd.read_csv(file, sep='\t', names=RAW_CONTEXT_DATA_NAMES)
    print(len(data))
    print(len(data))
    print(file_split)
    find_chars = ['。', '；', '，', '、', ';', ',', '？', ' ']
    cnt = 0
    truncate_context_with_placeholder = []
    for i, row in data.iterrows():
        context = row[RAW_CONTEXT_DATA_NAMES[-1]]
        if len(context) > max_len:
            idx = -1
            cnt += 1
            context = row[RAW_CONTEXT_DATA_NAMES[-1]][:truncate_len]
            for ch in find_chars:
                idx = context.rfind(ch)
                if idx > 0:
                    break
            if idx == -1:
                idx = truncate_len
            context = context[:idx] + '。' if PLH in context[
                                                    :idx] else f"以下是{PLH}的介绍：{context[:idx] + '。' if not context.endswith('）') else context[:idx + 1]}"
            print(f"{context} | {len(context)} | {idx}")
        if PLH not in context:
            print(f"{i} | {context}")
        assert len(context) <= truncate_len + len(PREFIX) + 1
        truncate_context_with_placeholder.append(context)
    data[RAW_CONTEXT_DATA_NAMES[-1]] = truncate_context_with_placeholder
    assert all(len(c) <= truncate_len + len(PREFIX) + 1 and PLH in c for c in data[RAW_CONTEXT_DATA_NAMES[-1]])
    print(f"Truncated: {cnt}")
    save_path = f'data/d1_ranker_truncate{truncate_len}_context_{file_split}.txt'
    print(data)
    print(f"Saved in: {save_path}")
    if save:
        data.to_csv(save_path, sep='\t', header=False, index=False)


def extract_context_sentence(file, min_len=75, truncate_len=150, save=False):
    """
    抽取context中PLH所在的关键句子作为最终的context，为中心扩展的思想，即:
    1. 通过标点符号将context转为一个句子的序列
    2. 定位PLH所在的句子
    3. 从该句子向两边扩展，当扩展到一定的长度区间后 [min_len, truncate_len] 后停止
    :param file: 带有PLH的context文件, 总共3列，[src(全称), target(缩略词), context]
    :param min_len: 中心扩展的最小长度
    :param truncate_len: 中心扩展的最大长度
    :param save: 是否保存
    :return: 保存处理的数据，格式和file一样，返回保存位置
    """
    file_split = thwpy.check_split(file)
    data = pd.read_csv(file, sep='\t', names=RAW_CONTEXT_DATA_NAMES)
    print(len(data))
    print(file_split)
    cnt = 0
    find_chars = ''.join(['。', '；', '，', '、', ';', ',', '？', '?', '！', '!', ' '])
    pattern = re.compile(rf'[{find_chars}]')
    truncate_context_with_placeholder = []
    for i, row in data.iterrows():
        context = row[RAW_CONTEXT_DATA_NAMES[-1]]
        context_sentences = re.split(pattern, context)
        context_sentences = [s for s in context_sentences if len(s) > 0]
        # print(context_sentences)
        pos = 0
        if PREFIX in context:
            len_sum = 0
            for idx, s in enumerate(context_sentences):
                len_sum += len(s)
                if len_sum > truncate_len:
                    pos = idx
                    break
            context = '，'.join(context_sentences[:pos] if pos > 0 else context_sentences)
        else:
            for idx, s in enumerate(context_sentences):
                if PLH in s:
                    context = s
                    offset = 1
                    while len(context) < min_len and (idx + offset <= len(context_sentences) or idx - offset >= 0):
                        context = '，'.join(
                            context_sentences[max(0, idx - offset): min(len(context_sentences), idx + offset)])
                        offset += 1
                    break
        assert PLH in context
        if PREFIX == context:
            cnt += 1
        print(f"{context} | {i} | {len(context)}")
        truncate_context_with_placeholder.append(context)
    assert len(truncate_context_with_placeholder) == len(data)
    print(
        f"Average Length: {sum(len(c) for c in truncate_context_with_placeholder) / len(truncate_context_with_placeholder)}")
    print(f"Max Length: {max(len(c) for c in truncate_context_with_placeholder)}")
    print(f"context = PREFIX: {cnt}")
    data[RAW_CONTEXT_DATA_NAMES[-1]] = truncate_context_with_placeholder
    save_path = f'data/d1_ranker_extract_truncate{truncate_len}_context_{file_split}.txt'
    print(data)
    print(f"Saved in {save_path}")
    if save:
        data.to_csv(save_path, sep='\t', header=False, index=False)
    return save_path


def read_candidates(file, top_k=12):
    """读取candidates文件file，并确保每个样本都有top_k个candidates"""
    data = pd.read_csv(file, sep='\t', names=['src', 'target', 'candidates'])
    data['candidates'] = data['candidates'].map(lambda x: x.split(';'))
    print(data['candidates'])
    for c in data['candidates']:
        assert len(c) == top_k


def remove_spans(context, abbr):
    """移除context中带有abbr的span"""
    if re.search(rf'（.*?{abbr}.*?）', context) is not None:
        context = re.sub(rf'（.*?{abbr}.*?）', '', context)
    while abbr in context:
        idx = context.find(abbr)
        start = idx - 1
        end = idx + len(abbr)
        punctuation = '，。；：！？,.;:!?'
        while start >= 0:
            if context[start] in punctuation:
                break
            start -= 1
        while end < len(context):
            if context[end] in punctuation:
                break
            end += 1
        if start < 0:
            context = context[end + 1:]
        elif end >= len(context):
            context = context[:start + 1]
        else:
            context = context[:start + 1] + context[end + 1:]
    return context


def prepare_context_with_placeholder(file, save=False):
    """
    给context文件加入一列，为带有PLH的context，方便填入全称和candidates
    :param file: context文件，包含3列 [src(全称), target(缩略词), context]
    :param save: 是否保存文件
    :return: 保存处理后的context数据，返回保存路径
    """
    data = pd.read_csv(file, sep='\t', names=RAW_CONTEXT_DATA_NAMES)
    file_split = thwpy.check_split(file)
    cnt = 0
    abbr_in_contexts = []
    context_with_placeholder = []
    for idx, row in data.iterrows():
        context = row['context']
        if row['src'] not in row['context']:
            cnt += 1
            context = f"以下是{PLH}的介绍：{context}"
        else:
            context = row['context'].replace(row['src'], PLH)
        if row['target'] in context:
            context = remove_spans(context, row['target'])
            abbr_in_contexts.append([row['target'], row['context'], context])
        if PLH not in context:
            context = f"以下是{PLH}的介绍：{context}"
        assert row['target'] not in context and PLH in context
        context_with_placeholder.append(context)
    print(f"Not in the context: {cnt}")
    # data['context_with_placeholder'] = context_with_placeholder
    data['context'] = context_with_placeholder
    print(data)
    print(data.keys())
    save_path = f'data/d1_ranker_context_{file_split}.txt'
    print(f"Saved in {save_path}")
    if save:
        data.to_csv(save_path, sep='\t', header=False, index=False)
    return save_path


def check_ablation(file_name):
    for ablation in ABLATIONS:
        if ablation in file_name:
            return ablation
    return ABLATIONS[0]


def prepare_label(file, top_k=12, has_duplicate=False, version=1, save=False):
    """
    找到candidates中的正样本，构建label
    对于训练数据，如果candidates中不存在ground truth，那么将ground truth替换candidates中的最后一个
    对于测试数据，如果candidates中不存在ground truth，那么将label设为-1
    :param save: 是否保存
    :param file: 带有candidates的文件
    :param top_k: 最终共有几个candidates
    :param has_duplicate: candidate中是否有重复
    :return: 保存添加了 label 列的数据
    """
    data = pd.read_csv(file, sep='\t', names=['src', 'target', 'candidates'])
    context = '_context' if 'context' in file else ''
    ratio = thwpy.regex_match(rf'_ratio\d*', file)
    gen_context_truncate = thwpy.regex_match(rf'_truncate\d*', file)
    file_split = thwpy.check_split(file)
    ablation = check_ablation(file)
    print(file_split)
    cnt = 0
    labels = []
    print(has_duplicate)
    for idx, row in data.iterrows():
        candidate_list = row['candidates'].split(';')
        if has_duplicate:
            tmp_list = list(set(candidate_list))
            tmp_list.sort(key=candidate_list.index)
            candidate_list = tmp_list[:top_k]
        if row['target'] in candidate_list:
            label = candidate_list.index(row['target'])
        else:
            cnt += 1
            label = top_k - 1 if file_split == 'train' else -1
            if file_split == 'train':
                candidate_list[-1] = row['target']

        assert len(candidate_list) == top_k, "candidate个数必须正确"
        assert (label != -1 and candidate_list[label] == row['target']) or (row['target'] not in candidate_list and label == -1), \
            "标签对应，且测试集中不存在ground truth的样本label为-1"

        row['candidates'] = ';'.join(candidate_list)
        labels.append(label)
    data['label'] = labels
    print(data)
    print(data.keys())
    print(f"Hit@{top_k}: {len(data) - cnt} / {len(data)} = {(len(data) - cnt) / len(data)}")
    save_path = f'data/d1_v{version}_candidate{ratio}{context}{gen_context_truncate}_{ablation}_top{top_k}_w_label_{file_split}.txt'  # d1
    print(f"Saved in {save_path}")
    if save:
        data.to_csv(save_path, sep='\t', header=False, index=False)
    return save_path, (len(data) - cnt) / len(data)


def merge_data(context_file, candidate_file, top_k=12, truncate_len=150, version=1, save=False):
    """
    将context文件和candidate文件合并，构造最终模型训练和测试读取的数据文件
    :param version:
    :param save: 是否保存
    :param context_file: 格式为['full', 'abbr', 'context']
    :param candidate_file: 格式为['full', 'abbr', 'candidates', 'label']
    :param top_k: 有几个candidates
    :param truncate_len: 超过truncate_len进行truncate
    :return: 保存merge后的数据
    """
    file_split = thwpy.check_split(context_file)
    candidate_file_split = thwpy.check_split(candidate_file)
    post_edit = f'_{check_ablation(candidate_file)}'
    extract = '_extract' if 'extract' in context_file else ''

    assert file_split == candidate_file_split, 'merge同一个split'

    context_data = pd.read_csv(context_file, sep='\t', names=RAW_CONTEXT_DATA_NAMES)
    max_len = max([len(c) for c in context_data[RAW_CONTEXT_DATA_NAMES[-1]]])
    print(f"max context len: {max_len}")
    candidate_data = pd.read_csv(candidate_file, sep='\t', names=CANDIDATE_WITH_LABEL_DATA_NAMES)

    assert len(context_data) == len(candidate_data)

    data = pd.merge(context_data, candidate_data, how='right', on=RAW_CONTEXT_DATA_NAMES[:2])
    print(data)
    print(data.keys())
    save_path = f'data/d1_v{version}_ranker{extract}{post_edit}_truncate{truncate_len}_top{top_k}_{file_split}.txt'  # d1
    print(f"Saved in {save_path}")
    if save:
        data.to_csv(save_path, sep='\t', header=False, index=False)
    return save_path


def get_subsequence_by_label(word, label):
    """
    根据label选取word中的元素组成一个word的子序列
    :param word:
    :param label: 0/1序列，str/list
    :return: 获取到的子序列，str类型
    """
    return ''.join([ch for ch, l in zip(word, label) if l == 1])


def sample_by_edit_distance(word, label, d=1):
    """
    根据和label的编辑距离d从word中选取子序列
    :param word:
    :param label: 0/1序列，str/list
    :param d: 最终的子序列和label的距离
    :return: 获取的子序列列表
    """
    label = [int(l) for l in label]
    # print(label)
    assert len(label) == len(word)

    idx_list = list(range(len(label)))
    indices = list(combinations(idx_list, d))
    random.shuffle(indices)
    candidates = []
    for idx in indices:
        current_label = label.copy()
        for i in idx:
            current_label[i] = 1 - current_label[i]
        candidate = get_subsequence_by_label(word, current_label)
        if len(candidate) > 0:
            candidates.append(candidate)
    return candidates


def sample_by_word_segmentation(seg_word, d=1):
    """
    选取和分词后的结果seg_word的编辑距离为d的子序列
    :param seg_word: 分词后的结果 list
    :param d: 编辑距离
    :return: 子序列的列表
    """
    labels = [1] * len(seg_word)
    idx_list = list(range(len(labels)))
    indices = list(combinations(idx_list, d))
    random.shuffle(indices)
    candidates = []
    for idx in indices:
        current_label = labels.copy()
        for i in idx:
            current_label[i] = 1 - current_label[i]
        candidate = get_subsequence_by_label(seg_word, current_label)
        if len(candidate) > 0:
            candidates.append(candidate)
    return candidates


def prepare_candidates(label_seq_file, candidate_file, top_k=12, version=1, save=False,
                       no_full=False, no_word=False, no_truth=False, no_char=False):
    """
    对candidate_file中模型生成的candidates通过启发式的规则进行后处理
    首先，对于训练集，首先全称是一个负样本
    1st. 根据全称的分词结果，选取和全称词编辑距离为d的子序列
    2nd. 只是对于训练集而言，选取和ground truth编辑距离为d的子序列
    3th. 选取和全称字符编辑距离为d的子序列，其中 0 < d < word_len
    :param label_seq_file: 带有0/1序列的文件
    :param candidate_file: 带有candidates的文件
    :param top_k: 最终保证top_k个candidates
    :param version:
    :param no_char: 是否移除 char-based rule
    :param no_truth: 是否移除 ground-truth-based rule
    :param no_word: 是否移除 word-based rule
    :param no_full: candidates是否移除全称，调用时默认设为True
    :param save: 是否保存
    :return: 保存最终的candidate数据的路径，以及Hit@K指标
    """
    file_split = thwpy.check_split(candidate_file)
    print(file_split)
    print(thwpy.check_split(label_seq_file))
    assert file_split == thwpy.check_split(label_seq_file) or file_split == 'infer'
    train = file_split == 'train'
    label_data = thwpy.load_csv(label_seq_file, sep='\t')
    label_data = [row[1] for row in label_data]
    candidate_data = pd.read_csv(candidate_file, sep='\t', names=['full', 'abbr', 'candidate'])

    candidate_data['label'] = label_data

    assert not train or all(len(row['full']) == len(row['label']) for i, row in candidate_data.iterrows())

    statistics = {'word': 0, 'ground-truth': 0, 'char': 0, 'pad': 0}
    candidates = []
    diff = 0
    edit_distances = []
    cnt = 0
    for idx, row in candidate_data.iterrows():
        diff += len(row['full']) - len(row['abbr'])
        edit_distances.append(len(row['full']) - len(row['abbr']))
        candidate_list = row['candidate'].split(';')

        assert len(candidate_list) == top_k

        tmp_list = list(set(candidate_list))
        tmp_list.sort(key=candidate_list.index)
        tmp_list = [c for c in tmp_list if is_subsequence(c, row['full']) and c != '']
        if not no_full:
            if train and len(tmp_list) < top_k and row['full'] not in tmp_list:  # d1
                tmp_list.append(row['full'])
        else:
            tmp_list = [c for c in tmp_list if c != row['full']]
        word_len = len(row['full'])
        seg_word = jieba.lcut(row['full'])
        seg_word_len = len(seg_word)

        # 1st
        if not no_word:
            d = 1
            while len(tmp_list) < top_k and d <= seg_word_len:
                examples = sample_by_word_segmentation(seg_word, d)
                for example in examples:
                    if example not in tmp_list:
                        tmp_list.append(example)
                        statistics['word'] += 1
                        if len(tmp_list) == top_k:
                            break
                d += 1
        # 2nd
        if (not no_truth) and train:
            d = 1
            while len(tmp_list) < top_k and d <= word_len:
                examples = sample_by_edit_distance(row['full'], row['label'], d)
                for example in examples:
                    if example not in tmp_list:
                        tmp_list.append(example)
                        statistics['ground-truth'] += 1
                        if len(tmp_list) == top_k:
                            break
                d += 1
        # 3rd
        if not no_char:
            d = 1
            while len(tmp_list) < top_k and d < word_len:
                examples = sample_by_word_segmentation(row['full'], d)
                for example in examples:
                    if example not in tmp_list:
                        tmp_list.append(example)
                        statistics['char'] += 1
                        if len(tmp_list) == top_k:
                            break
                d += 1
        # len(word) <= 3，枚举肯定会有空的，需要补全

        # v1
        if no_full:
            tmp_list = [c for c in tmp_list if c != row['full'] and c != '']
            d = len(tmp_list) - 1
            while d >= 0 and tmp_list[d] == row['abbr']:
                d -= 1
            while d >= 0 and len(tmp_list) < top_k:
                tmp_list.append(tmp_list[d])
                statistics['pad'] += 1
        else:
            while len(tmp_list) < top_k:
                tmp_list.append(row['full'])

        if no_full:
            assert row['full'] not in tmp_list and '' not in tmp_list
        # padding
        while len(tmp_list) < top_k:
            tmp_list.append(row['full'])
            statistics['pad'] += 1
        assert len(tmp_list) == top_k
        if row['abbr'] in tmp_list:
            cnt += 1
        candidates.append(';'.join(tmp_list))
        print(f"{row['full']} | {row['abbr']} | {';'.join(tmp_list)}")
    print(f"Split: {file_split}")
    print(f"Average Edit Distance: {diff / len(candidate_data)}")
    edit_distances.sort()
    print(f"Middle Edit Distance: {edit_distances[len(candidate_data) // 2]}")
    print(f"Hit@{top_k}: {cnt / len(candidate_data)}")
    print(f"Statistics: {statistics}")
    candidate_data['candidate'] = candidates

    candidate_data = candidate_data.drop(columns=['label'])

    print(candidate_data)
    print(candidate_data.keys())

    if no_word:
        post_edit = 'noword'
    elif no_char:
        post_edit = 'nochar'
    elif no_truth:
        post_edit = 'notruth'
    else:
        post_edit = 'all'
    save_path = f'data/d1_v{version}_candidate_{post_edit}_top{top_k}_{file_split}.txt'

    print(f"Saved in {save_path}")
    if save:
        candidate_data.to_csv(save_path, sep='\t', header=False, index=False)
    return save_path, cnt / len(candidate_data)


def generate_candidates_by_rule(full, candidate_list, no_word=False, no_char=False):
    top_k = len(candidate_list)
    tmp_list = list(set(candidate_list))
    tmp_list.sort(key=candidate_list.index)
    tmp_list = [c for c in tmp_list if is_subsequence(c, full) and c != '']
    # tmp_list = [c for c in tmp_list if is_subsequence(c, row['full'])]
    tmp_list = [c for c in tmp_list if c != full]
    word_len = len(full)
    seg_word = jieba.lcut(full)
    seg_word_len = len(seg_word)

    # 1st
    if not no_word:
        d = 1
        while len(tmp_list) < top_k and d <= seg_word_len:
            examples = sample_by_word_segmentation(seg_word, d)
            for example in examples:
                if example not in tmp_list:
                    tmp_list.append(example)
                    if len(tmp_list) == top_k:
                        break
            d += 1
    if not no_char:
        d = 1
        while len(tmp_list) < top_k and d < word_len:
            examples = sample_by_word_segmentation(full, d)
            for example in examples:
                if example not in tmp_list:
                    tmp_list.append(example)
                    if len(tmp_list) == top_k:
                        break
            d += 1
    # len(word) <= 3，枚举肯定会有空的，需要补全

    # v1
    tmp_list = [c for c in tmp_list if c != full and c != '']
    # tmp_list = [c for c in tmp_list if c != row['full']]
    d = len(tmp_list) - 1
    while d >= 0 and len(tmp_list) < top_k:
        tmp_list.append(tmp_list[d])

    while len(tmp_list) < top_k:
        tmp_list.append(full)
    assert len(tmp_list) == top_k
    return tmp_list


def load_dataset(file, to_dataset=True):
    fulls, abbrs = read_text_pairs(file)
    fulls = [str(s) for s in fulls]
    abbrs = [str(s) for s in abbrs]
    if to_dataset:
        return Dataset.from_dict({'full': fulls, 'abbr': abbrs})
    return fulls, abbrs


def save_text_pairs(src, target, file):
    pd.DataFrame({'src': src, 'target': target}).to_csv(file, sep='\t', header=False, index=False)


def split_pretrain_data(file, ratio=0.01):
    long, short = read_text_pairs(file)
    idx = list(range(len(long)))
    random.shuffle(idx)
    val_len = int(len(long) * ratio)
    print(val_len)
    print(len(long) - val_len)
    val_idx = idx[:val_len]
    train_idx = idx[val_len:]
    long, short = np.array(long), np.array(short)
    save_text_pairs(long[train_idx], short[train_idx], f"{file.split('.')[0]}_train.txt")
    save_text_pairs(long[val_idx], short[val_idx], f"{file.split('.')[0]}_val.txt")


def compare(file1, file2):
    l1 = []
    with open(file1, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            l1.append(line)
            if line == '':
                print(i + 1)
    print(len(l1))
    print(l1)
    _, abbrs = read_text_pairs(file2)
    hit = 0
    print(len(abbrs))
    for i, j in zip(l1, abbrs):
        if i != j:
            print(f"{i}, {j}")
        else:
            hit += 1
    print(hit)
    print(len(abbrs))
    print((hit + 2) / (len(abbrs) - 2))


def prepare_m2e_pairs(m2e_file):
    mentions, entities = read_text_pairs(m2e_file)
    cnt = 0
    for i, (m, e) in enumerate(zip(mentions, entities)):
        m = str(m)
        e = str(e)
        mentions[i] = m.strip().replace(' ', '')
        entities[i] = e.strip().replace(' ', '')
        if len(mentions[i]) < len(entities[i]):
            tmp = mentions[i]
            mentions[i] = entities[i]
            entities[i] = tmp
            cnt += 1
            print(f"Line {i + 1}: {mentions[i]}\t{entities[i]}")
    pd.DataFrame({'long': mentions, 'short': entities}).to_csv('data/pretrain.txt', sep='\t', header=False, index=False)


def is_subsequence(s, t):
    """判断s是否为t的子序列"""
    i = 0
    for ch in s:
        while i < len(t) and t[i] != ch:
            i += 1
        if i >= len(t):
            return False
        i += 1
    return True


def is_available(mention, entity):
    return re.search(r'\(.*\)', mention) is None and re.search(r'（.*）', mention) is None \
        # and is_subsequence(entity, mention)


def find_bracket(s):
    match = re.search(r'（.*）', s)
    if match is None:
        match = re.search(r'\(.*\)', s)
    return match


def check_subsequence_candidates(file):
    data = pd.read_csv(file, sep='\t', names=['full', 'abbr', 'candidate', 'label'])
    fulls, abbrs, candidates = data['full'].tolist(), data['abbr'].tolist(), [x.split(';') for x in data['candidate']]
    cnt = 0
    for full, abbr, candidate in zip(fulls, abbrs, candidates):
        if len(full) <= 5:
            cnt += 1
            print(f"{full}\t{abbr}\t{candidate}")
    print(cnt)


def common_prefix(s, t):
    for i, (s_ch, t_ch) in enumerate(zip(s, t)):
        if s_ch != t_ch:
            return s[:i].rstrip('_')
    return s.split('.')[0]


def avoid_leakage(file):
    data = pd.read_csv(file, sep='\t', names=RAW_CONTEXT_DATA_NAMES)
    for i, row in data.iterrows():
        assert row['target'] not in row['context']
    print("NO LEAKAGE")


def check_pretrain_leakage(pretrain_file, abbr_file, save=False):
    pretrain_data = pd.read_csv(pretrain_file, sep='\t', names=GEN_DATA_NAMES)
    data = pd.read_csv(abbr_file, sep='\t', names=GEN_DATA_NAMES)
    data_dict = {k: v for k, v in zip(data[GEN_DATA_NAMES[1]], data[GEN_DATA_NAMES[0]])}
    cnt = 0
    remove_idx = []
    for i, row in pretrain_data.iterrows():
        if row[GEN_DATA_NAMES[1]] in data_dict and row[GEN_DATA_NAMES[0]] == data_dict[row[GEN_DATA_NAMES[1]]]:
            cnt += 1
            remove_idx.append(i)
    print(len(pretrain_data))
    print(cnt)
    print("After Filtering: ")
    pretrain_data = pretrain_data.drop(remove_idx).reset_index(drop=True)
    print(len(pretrain_data))
    print(pretrain_data)
    if save:
        pretrain_data.to_csv('data/pretrain_noleak_train.txt', sep='\t', header=False, index=False)


def search_ranker_data(file, query):
    data = pd.read_csv(file, sep='\t', names=RANKER_DATA_NAMES)
    for idx, row in data.iterrows():
        if row[RANKER_DATA_NAMES[0]] == query:
            pprint(f"context w [PLH]: {row[RANKER_DATA_NAMES[-3]]}")
            pprint(f"candidates: {row[RANKER_DATA_NAMES[-2]]}")
            break


def search_candidate_data(file, query):
    data = pd.read_csv(file, sep='\t', names=CANDIDATE_DATA_NAMES)
    for idx, row in data.iterrows():
        if row[CANDIDATE_DATA_NAMES[0]] == query:
            pprint(f"candidates: {row[CANDIDATE_DATA_NAMES[-1]]}")
            break


def get_label_seq(full, abbr):
    label = [0] * len(full)
    idx = 0
    if len(abbr) == 0:
        return label
    for i, ch in enumerate(full):
        if ch == abbr[idx]:
            label[i] = 1
            idx += 1
            if idx == len(abbr):
                break
    return label


def check_identical(file1, file2):
    data1 = pd.read_csv(file1, sep='\t', names=CANDIDATE_DATA_NAMES)
    data2 = pd.read_csv(file2, sep='\t', names=CANDIDATE_DATA_NAMES)
    for i, (d1, d2) in enumerate(zip(data1[CANDIDATE_DATA_NAMES[-1]], data2[CANDIDATE_DATA_NAMES[-1]])):
        if set(d1) != set(d2):
            print(i)
            print(d1)
            print(d2)


def count_no_subsequence(file):
    data = pd.read_csv(file, sep='\t', names=CANDIDATE_DATA_NAMES)
    cnt = 0
    total = 0
    for i, row in data.iterrows():
        candidates = row[CANDIDATE_DATA_NAMES[-1]].split(';')
        total += len(candidates)
        for c in candidates:
            if not is_subsequence(c, row[CANDIDATE_DATA_NAMES[0]]):
                cnt += 1
    print(f"{cnt} / {total} = {cnt / total}")


def sample_predictions_with_scores(predict_file, context_predict_file, num=300, save=False):
    names = CANDIDATE_DATA_NAMES + ['score']
    predicts = pd.read_csv(predict_file, names=names, sep='\t')
    context_predicts = pd.read_csv(context_predict_file, names=names, sep='\t')

    assert len(predicts) == len(context_predicts)

    indices = list(range(len(predicts)))
    random.shuffle(indices)
    indices = indices[:num]
    predicts = predicts.iloc[indices]
    context_predicts = context_predicts.iloc[indices]
    print(predicts)
    merge = []
    for (i, row), (j, context_row) in zip(predicts.iterrows(), context_predicts.iterrows()):
        merge.append({
            'idx': i,
            'full': row[CANDIDATE_DATA_NAMES[0]],
            'abbreviation': row[CANDIDATE_DATA_NAMES[1]],
            'candidate': row[CANDIDATE_DATA_NAMES[2]],
            'context_candidate': context_row[CANDIDATE_DATA_NAMES[2]],
            'score': row[names[-1]],
            'context_score': context_row[names[-1]],
            'label': -1
        })
    pprint(merge)
    save_path = f'eval/sample_{num}_compare_score.json'
    print(f"Saved in {save_path}")
    if save:
        json.dump(merge, open(save_path, 'w+', encoding='utf-8'), ensure_ascii=False, indent=4)


def construct_ranker_context(context_file, save=False):
    """
    构建用于ranker的context数据
    :param context_file: raw context文件，包含3列 [src(全称), target(缩略词), context]
    :param save: 是否保存
    :return: 保存的context数据路径
    """
    context_with_placeholder_save_path = prepare_context_with_placeholder(file=context_file, save=save)
    extracted_context_save_path = extract_context_sentence(context_with_placeholder_save_path, truncate_len=150, save=save)
    return extracted_context_save_path


def construct_ranker_dataset(label_seq_file, candidate_file, context_file, top_k=12, version=1, save=False,
                             no_word=False, no_truth=False, no_char=False):
    """
    构建ranker的数据集
    :param label_seq_file: 带有0/1序列的文件，2列[src(全称), label(对应缩略词的0/1序列)]
    :param candidate_file: 带有candidates的文件，3列[src(全称), target(缩略词), candidates]
    :param context_file: 需要合并的context文件，3列[src(全称), target(缩略词), context]
    :param top_k: candidates个数，默认12
    :param version:
    :param save: 是否保存
    :param no_char: 是否移除 char-based rule
    :param no_truth: 是否移除 ground-truth-based rule
    :param no_word: 是否移除 word-based rule
    :return: 最终保存的数据集的路径
    """
    split = thwpy.check_split(label_seq_file)
    assert split == thwpy.check_split(candidate_file) and split == thwpy.check_split(context_file), '数据集划分不同'
    candidate_save_path, candidate_hit_k = prepare_candidates(label_seq_file=label_seq_file,
                                                              candidate_file=candidate_file,
                                                              version=version,
                                                              top_k=top_k,
                                                              no_full=True,
                                                              no_word=no_word, no_truth=no_truth, no_char=no_char,
                                                              save=save)
    candidate_w_label_save_path, label_hit_k = prepare_label(candidate_save_path, top_k=top_k, version=version,
                                                             save=save)

    assert candidate_hit_k == label_hit_k

    dataset_save_path = merge_data(context_file,
                                   candidate_w_label_save_path,
                                   top_k=top_k,
                                   truncate_len=150,
                                   version=version,
                                   save=save)
    return dataset_save_path


if __name__ == '__main__':
    # val, train, test
    # prepare context
    extracted_context_path = construct_ranker_context(context_file='data/d1_raw_context_test.txt', save=True)
    construct_ranker_dataset(label_seq_file='data/d1_test.txt',
                             candidate_file='eval/d1_cpt_pretrain_noleak_128_accum_1_10_128_10_best_test_beam_32_candidates_12.txt',
                             context_file=extracted_context_path,
                             top_k=12,
                             version=1,
                             save=True)

    # avoid_leakage('data/d1_ranker_extract_truncate150_context_train.txt')
    # avoid_leakage('data/d1_ranker_extract_truncate150_context_val.txt')
    # avoid_leakage('data/d1_ranker_extract_truncate150_context_test.txt')