from transformers import BertTokenizer, BertConfig
import os
import torch

generation_model_types = ['cpt', 'bart']


class Config:
    def __init__(self, batch_size=128, num_epochs=10, lr=2e-5, dataset='d1',
                 model_name='cpt-base', wandb=False):
        self.current_path = os.path.dirname(__file__)
        self.dataset = dataset
        self.train_path = os.path.join(self.current_path, f'data/{self.dataset}_gen_train.txt')
        self.val_path = os.path.join(self.current_path, f'data/{self.dataset}_gen_val.txt')
        self.test_path = os.path.join(self.current_path, f'data/{self.dataset}_gen_test.txt')
        self.model_name = model_name
        self.model_type = model_name
        for model_type in generation_model_types:
            if model_type in model_name:
                self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.src_max_len = 32
        self.target_max_len = 20
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_rate = 0.3
        self.ignore_pad_token_for_loss = True
        self.learning_rate = lr
        self.save_path = f"{self.dataset}_{self.model_name.replace('/', '_')}_{self.batch_size}_{self.num_epochs}"
        self.best_model_path = f"{self.save_path}_best"
        self.logging_file_name = f"{self.dataset}_{self.model_name.replace('/', '_')}_logging_{self.batch_size}_{self.num_epochs}.log"
        self.predict_file = os.path.join(self.current_path, f'eval/{self.save_path}.txt')
        self.tensorboard_dir = f'runs/{self.logging_file_name.split(".")[0]}'
        self.wandb = wandb
        if not wandb:
            os.environ["WANDB_DISABLED"] = "true"


class PretrainConfig:
    def __init__(self, batch_size=128, num_epochs=10, lr=2e-5, gradient_accumulation_steps=1,
                 model_name='cpt-base', wandb=False):
        self.current_path = os.path.dirname(__file__)
        self.setting = '_noleak'
        self.post_process = ''
        self.train_path = os.path.join(self.current_path, f'data/pretrain{self.setting}_train{self.post_process}.txt')
        self.val_path = os.path.join(self.current_path, f'data/pretrain_val{self.post_process}.txt')
        self.model_name = model_name
        self.model_type = model_name
        for model_type in generation_model_types:
            if model_type in model_name:
                self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)  # CPT, BART
        self.src_max_len = 32
        self.target_max_len = 20
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_rate = 0.3
        self.ignore_pad_token_for_loss = True
        self.learning_rate = lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_accumulation_steps = gradient_accumulation_steps
        self.save_path = f"{self.model_type}_pretrain{self.setting}{self.post_process}_{self.batch_size}_accum_{self.gradient_accumulation_steps}_{self.num_epochs}"
        self.best_model_path = f"{self.save_path}_best"
        self.train_set_cache_path = os.path.join(self.current_path, f'data/pretrain_cache{self.setting}_train{self.post_process}')
        self.val_set_cache_path = os.path.join(self.current_path, f'data/pretrain_cache{self.setting}_val{self.post_process}')
        self.logging_file_name = f'{self.model_type}_pretrain{self.setting}{self.post_process}_logging_{self.batch_size}_accum_{self.gradient_accumulation_steps}_{self.num_epochs}.log'
        self.tensorboard_dir = f'runs/{self.logging_file_name.split(".")[0]}'
        self.predict_file = os.path.join(self.current_path, f'data/{self.save_path}.txt')
        self.wandb = wandb
        if not wandb:
            os.environ["WANDB_DISABLED"] = "true"


class RankerConfig:
    def __init__(self, batch_size=4, num_epochs=10, lr=4e-5, mode='add_b', gradient_accumulation_steps=512, dataset='d1',
                 ablation='all', top_k=12, truncation=150, no_extract=False, version=1, wandb=False):
        self.current_path = os.path.dirname(__file__)
        self.dataset = dataset
        self.num_candidates = top_k
        self.truncate_len = truncation
        self.ABLATIONS = ['_notruth', '_nochar',  '_noword', '_noheu', '_all']
        self.MODES = ['add_b', 'none']
        self.post_edit = f"_{ablation}"
        self.version = f"_v{version}" if version > 0 else ''
        assert self.post_edit in self.ABLATIONS
        self.extract = "" if no_extract else "_extract"  # ["_extract", ""]
        assert mode in self.MODES
        self.mode = mode
        self.train_path = os.path.join(self.current_path, f'data/{self.dataset}{self.version}_ranker{self.extract}{self.post_edit}_truncate{self.truncate_len}_top{self.num_candidates}_train.txt')
        self.val_path = os.path.join(self.current_path, f'data/{self.dataset}{self.version}_ranker{self.extract}{self.post_edit}_truncate{self.truncate_len}_top{self.num_candidates}_val.txt')
        self.test_path = os.path.join(self.current_path, f'data/{self.dataset}{self.version}_ranker{self.extract}{self.post_edit}_truncate{self.truncate_len}_top{self.num_candidates}_test.txt')
        self.model_name = os.path.join(self.current_path, 'chinese-macbert-base')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_config = BertConfig.from_pretrained(self.model_name)
        self.max_len = 512
        self.temp = 0.05
        self.hidden_size = self.bert_config.hidden_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_rate = 0.3
        self.clip_norm = 1.0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = 0.01
        self.learning_rate = lr
        self.save_path = f"{self.dataset}{self.version}_{str(lr).split('-')[0]}_ranker{self.extract}{self.post_edit}_truncate{self.truncate_len}_top{self.num_candidates}_{mode}_{self.batch_size}_accu_{self.gradient_accumulation_steps}_{self.num_epochs}.pth"
        self.logging_file_name = f"{self.dataset}{self.version}_{str(lr).split('-')[0]}_ranker{self.extract}{self.post_edit}_truncate{self.truncate_len}_top{self.num_candidates}_{mode}_logging_{self.batch_size}_accu_{self.gradient_accumulation_steps}_{self.num_epochs}.log"
        self.wandb = wandb
        if not wandb:
            os.environ["WANDB_DISABLED"] = "true"


if __name__ == '__main__':
    config = RankerConfig(dataset='d1', top_k=12)
    print(config.train_path)
    print(config.num_candidates)
    print(config.logging_file_name)
    print(os.path.exists(config.train_path))
    print(os.path.exists(config.val_path))
    print(os.path.exists(config.test_path))

    config = Config(model_name='cpt_pretrain_noleak_128_accum_1_10', dataset='d1')
    print(config.train_path)
    print(config.save_path)
    print(config.logging_file_name)
    print(config.model_type)
    print(config.best_model_path)
    print(os.path.exists(config.train_path))
    print(os.path.exists(config.val_path))
    print(os.path.exists(config.test_path))


