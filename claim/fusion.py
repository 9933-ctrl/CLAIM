"""Context span + TransE embedding MLP category head (auxiliary to CRF)."""
import torch
from torch import nn
from .config import CATEGORIES


class KGFusionClassifier(nn.Module):
    def __init__(self, context_dim=512, kg_dim=256, hidden=256):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(context_dim+kg_dim,hidden),
                                     nn.ReLU(),nn.Dropout(.1),nn.Linear(hidden,len(CATEGORIES)))

    def forward(self, span_features, kg_features):
        if span_features.ndim != 2 or kg_features.ndim != 2:
            raise ValueError('Inputs must be [num_spans,features] tensors')
        return self.network(torch.cat((span_features,kg_features),dim=-1))


def collect_gold_spans(features, records, char_indices, kg_store, device):
    """Make trainable gold-span features. char_indices maps CRF character order to text offsets."""
    contexts, kgs, labels = [], [], []
    for batch_i, record in enumerate(records):
        idx_to_pos = {int(char):j for j,char in enumerate(char_indices[batch_i]) if int(char)>=0}
        for span in record.get('spans',[]):
            positions = [idx_to_pos[i] for i in range(span['start'],span['end']) if i in idx_to_pos]
            if not positions:
                continue
            contexts.append(features[batch_i,positions].mean(0))
            term = record['text'][span['start']:span['end']]
            kgs.append(kg_store.embedding(term).to(device))
            labels.append(CATEGORIES.index(span['label']))
    if not contexts:
        return None
    return torch.stack(contexts),torch.stack(kgs),torch.tensor(labels,device=device)
