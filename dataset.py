from torch.utils.data import Dataset, DataLoader
from config import RankerConfig
from transformers import DataCollatorWithPadding
from preprocess import read_ranker_data
from pprint import pprint
import torch


class RankerDataset(Dataset):
    def __init__(self, references, contexts, labels, config):
        super(RankerDataset, self).__init__()
        self.config = config
        self.references = references
        self.contexts = contexts
        self.labels = labels
        self.data_collator = DataCollatorWithPadding(tokenizer=config.tokenizer, padding=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ref_encoded = self.config.tokenizer(self.references[index], padding=False, truncation=True, max_length=self.config.max_len - 2)
        context_encoded = self.config.tokenizer(self.contexts[index], padding=False, truncation=True, max_length=self.config.max_len - 2)
        return {'reference': ref_encoded, 'context': context_encoded, 'label': self.labels[index]}

    def collate_fn(self, data):
        ref_encoded = [item['reference'] for item in data]
        keys = ['input_ids', 'token_type_ids', 'attention_mask']
        context_encoded = {k: [c for item in data for c in item['context'][k]] for k in keys}
        assert all(len(input_ids) == len(token_type_ids) == len(attention_mask)
                   for input_ids, token_type_ids, attention_mask in zip
                   (context_encoded['input_ids'], context_encoded['token_type_ids'], context_encoded['attention_mask']))
        ref_encoded = self.data_collator(ref_encoded)
        context_encoded = self.data_collator(context_encoded)
        labels = torch.tensor([item['label'] for item in data], dtype=torch.long)
        return {
            'reference': ref_encoded,
            'context': context_encoded,
            'labels': labels
        }


if __name__ == '__main__':
    config = RankerConfig()
    references, contexts, labels = read_ranker_data('data/d1_v1_ranker_extract_all_truncate150_top12_test.txt')
    dataset = RankerDataset(references, contexts, labels, config)
    # collator = DataCollatorWithPadding(config.tokenizer, padding=True)
    # print({k: v.size() for k, v in batch.items()})
    data_loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
    batch = next(iter(data_loader))
    print(batch['reference'].keys())
    print(batch['reference']['input_ids'].size())
    print(batch['reference']['attention_mask'].size())
    print(batch['reference']['token_type_ids'].size())
    print(batch['context']['input_ids'].size())
    print(batch['context']['attention_mask'].size())
    print(batch['context']['token_type_ids'].size())
    print(len(batch['labels']))