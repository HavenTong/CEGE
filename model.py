import torch
import torch.nn as nn
from transformers import BertModel


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        return self.activation(self.fc(inputs))


class Similarity(nn.Module):
    def __init__(self, temp=0.05):
        super(Similarity, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.temp = temp

    def forward(self, x, y):
        return self.cos_sim(x, y) / self.temp


class Ranker(nn.Module):
    def __init__(self, config):
        super(Ranker, self).__init__()
        self.config = config
        self.plm = BertModel.from_pretrained(config.model_name)
        self.similarity = Similarity()

    def forward(self, reference, context):
        ref_outputs = self.plm(**reference).pooler_output  # [batch_size, hidden_size]
        context_outputs = self.plm(**context).pooler_output  # [batch_size * num_candidates, hidden_size]

        batch_size = ref_outputs.size(0)
        hidden_size = ref_outputs.size(-1)

        context_outputs = context_outputs.reshape(-1, self.config.num_candidates, hidden_size)
        ref_outputs = ref_outputs.unsqueeze(1).repeat(1, self.config.num_candidates, 1)

        return self.similarity(context_outputs, ref_outputs)
