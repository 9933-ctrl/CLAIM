"""Character-aligned BIO preprocessing. JSONL offset convention is [start,end)."""
import json
import random
from pathlib import Path
from torch.utils.data import Dataset
from .config import TAG2ID


def read_jsonl(path):
    with open(path, encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def validate_record(record):
    text = record['text']
    occupied = set()
    for span in record.get('spans', []):
        start, end, label = span['start'], span['end'], span['label']
        if not 0 <= start < end <= len(text):
            raise ValueError(f'Invalid offsets {start}:{end} for {text!r}')
        if f'B-{label}' not in TAG2ID:
            raise ValueError(f'Unknown imagery category: {label}')
        if any(index in occupied for index in range(start, end)):
            raise ValueError('Overlapping spans are not supported by BIO single-label tagging')
        occupied.update(range(start, end))


def character_tags(record):
    validate_record(record)
    tags = ['O'] * len(record['text'])
    for span in record.get('spans', []):
        tags[span['start']] = f"B-{span['label']}"
        for position in range(span['start'] + 1, span['end']):
            tags[position] = f"I-{span['label']}"
    return tags


def encode_record(record, tokenizer, max_length):
    """Use individual characters as pre-tokenized words to retain original character offsets.

    Hugging Face fast tokenizer.word_ids() maps subwords back to the original characters.
    Special tokens and noninitial subword tokens have supervision mask=0.
    """
    chars = list(record['text'])
    if not chars:
        raise ValueError('Empty input poem')
    tokenized = tokenizer(chars, is_split_into_words=True, truncation=True,
                          max_length=max_length, padding='max_length',
                          return_attention_mask=True)
    tags = character_tags(record)
    word_ids = tokenized.word_ids()
    labels, mask, char_indices = [], [], []
    previous = None
    for word_id in word_ids:
        if word_id is not None and word_id != previous:
            labels.append(TAG2ID[tags[word_id]])
            mask.append(1)
            char_indices.append(word_id)
        else:
            labels.append(TAG2ID['O'])
            mask.append(0)
            char_indices.append(-1)
        previous = word_id
    # CRF operates on a compact sequence of supervised character positions.
    return {'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels, 'label_mask': mask, 'char_indices': char_indices}


class ImageryDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=128):
        self.records = read_jsonl(path)
        self.encodings = [encode_record(r, tokenizer, max_length) for r in self.records]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        import torch
        return {**{key: torch.tensor(value, dtype=torch.long)
                for key, value in self.encodings[index].items()}, 'record_index': torch.tensor(index)}


def split_by_poem(records, ratios=(.7,.15,.15), seed=42):
    """Poem-level random split; replace with iterative stratification if labels are imbalanced."""
    records = list(records)
    random.Random(seed).shuffle(records)
    a = int(len(records) * ratios[0]); b = int(len(records) * (ratios[0] + ratios[1]))
    return records[:a], records[a:b], records[b:]


def write_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
