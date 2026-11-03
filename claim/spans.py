"""Decode character-level BIO predictions into spans; malformed I tags begin a span."""
from .config import ID2TAG


def decode_spans(text, tag_ids, char_indices=None):
    if char_indices is None:
        char_indices = list(range(len(tag_ids)))
    results = []
    current = None
    for position, tag_id in zip(char_indices, tag_ids):
        if position < 0 or position >= len(text):
            continue
        tag = ID2TAG[int(tag_id)]
        if tag == 'O':
            if current is not None:
                results.append(current)
                current = None
            continue
        prefix, label = tag.split('-', 1)
        if current is None or prefix == 'B' or current['label'] != label or position != current['end']:
            if current is not None:
                results.append(current)
            current = {'start': position, 'end': position + 1, 'label': label}
        else:
            current['end'] = position + 1
    if current is not None:
        results.append(current)
    for span in results:
        span['text'] = text[span['start']:span['end']]
    return results
