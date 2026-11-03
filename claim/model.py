"""Fine-tuned BERT-wwm -> bidirectional LSTM -> CRF (33 BIO labels)."""
import torch
from torch import nn
from transformers import AutoModel
from .config import TAGS
from .crf import LinearCRF


class ImageryRecognizer(nn.Module):
    def __init__(self, backbone='hfl/chinese-bert-wwm', lstm_hidden=256, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, lstm_hidden, batch_first=True,
                            bidirectional=True)
        self.classifier = nn.Linear(2*lstm_hidden, len(TAGS))
        self.crf = LinearCRF(len(TAGS))

    def forward(self, input_ids, attention_mask, label_mask, labels=None):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Gather character-first embeddings into dense CRF sequences.
        lengths = label_mask.sum(1)
        max_chars = int(lengths.max().item())
        if max_chars < 1:
            raise ValueError('Tokenizer returned no usable characters')
        batch, _, dim = x.shape
        sequence = x.new_zeros((batch, max_chars, dim))
        packed_labels = torch.zeros((batch, max_chars), dtype=torch.long, device=x.device)
        crf_mask = torch.arange(max_chars, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        for b in range(batch):
            n = int(lengths[b])
            sequence[b, :n] = x[b, label_mask[b].bool()]
            if labels is not None:
                packed_labels[b, :n] = labels[b, label_mask[b].bool()]
        lstm_output, _ = self.lstm(self.dropout(sequence))
        emissions = self.classifier(self.dropout(lstm_output))
        loss = self.crf(emissions, packed_labels, crf_mask) if labels is not None else None
        return {'loss': loss, 'tags': self.crf.decode(emissions, crf_mask),
                'features': lstm_output, 'mask': crf_mask}
