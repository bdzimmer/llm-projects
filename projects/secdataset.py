"""
Create datasets from secondary files.
"""

from typing import Any, Dict, List, Tuple, Callable

import tqdm

from projects import secondary


Uid = Tuple[str, int, int]
Record = Tuple[Uid, str]


def sec_records_paragraphs(
        items: Dict[str, Dict],
        window_size: int,
        window_step: int
        ) -> Tuple[List[Record], List[Record], List[Dict]]:
    """
    Create a set of records from items by individual paragraph
    as well as sliding windows of a certain number of paragraphs.
    """

    records_paragraphs = []
    records_windows = []
    missing_notes = []

    for item_id, item in items.items():

        if 'notes' not in item:
            missing_notes.append(item['id'])
            continue

        notes = secondary.remove_tag_brackets(item['notes'])

        paragraphs = secondary.split_paragraphs(notes)
        for idx, paragraph in enumerate(paragraphs):
            uid = (
                item_id,
                idx
            )
            records_paragraphs.append((uid, paragraph))

        windows = windowed(paragraphs, window_size, window_size // 2)
        for w_idx, window in enumerate(windows):
            window_text = '\n\n'.join(window)
            uid = (
                item_id,
                w_idx * window_step,
                w_idx * window_size + len(window)
            )
            records_windows.append((uid, window_text))

    print('total paragraph records:', len(records_paragraphs))
    print('total window records', len(records_windows))
    print('missing notes:', missing_notes)

    return records_paragraphs, records_windows, missing_notes


def calculate_embeddings(records: List[Record], embs_func: Callable, batch_size: int) -> List[Dict[str, Any]]:
    """Calculate embeddings across a dataset in batches."""

    res = []

    # no overlap!
    for batch in tqdm.tqdm(windowed(records, batch_size, batch_size)):
        embs = embs_func([x for _, x in batch]).cpu().numpy()
        for idx, (uid, text) in enumerate(batch):
            em = embs[idx, :]
            record = {
                'uid': uid,
                'text': text,
                'emb': em
            }
            res.append(record)

    assert len(res) == len(records)

    return res


def windowed(xs: List, size: int, step: int) -> List[List]:
    """Find overlapping windows."""
    res = []
    for idx in range(0, len(xs), step):
        window = xs[idx:idx + size]
        res.append(window)
    return res
