"""Architecture and reproducibility configuration."""
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = ['Moon','Wind','Water','Flower','Jade','Sword','Lamp','Wine',
              'Grief','Longing','Exile','Joy','Dragon','Phoenix','Immortal','Spirit']
TAGS = ['O'] + [f'{prefix}-{cat}' for cat in CATEGORIES for prefix in ('B','I')]
TAG2ID = {tag: idx for idx, tag in enumerate(TAGS)}
ID2TAG = {idx: tag for tag, idx in TAG2ID.items()}

@dataclass
class Config:
    backbone: str = 'hfl/chinese-bert-wwm'
    max_length: int = 128
    lstm_hidden: int = 256
    kg_dim: int = 256
    dropout: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 20
    warmup_ratio: float = 0.1
    seed: int = 42
    max_grad_norm: float = 1.0
    lambda_gen: float = 0.3
    device: str = 'auto'
    output_dir: str = 'outputs'
