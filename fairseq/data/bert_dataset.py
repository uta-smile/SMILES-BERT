# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import random

import numpy as np
import torch

from fairseq import utils

from . import FairseqDataset, data_utils


def collate(samples,
            pad_idx,
            eos_idx,
            left_pad_source=False,
            left_pad_target=False):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)

    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target
    }
    return batch


class BertDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
    """

    def __init__(self,
                 ds,
                 ds_sizes,
                 dictionary,
                 left_pad_source=False,
                 left_pad_target=False,
                 max_source_positions=1024,
                 max_target_positions=1024,
                 shuffle=True,
                 pad_go=True):
        self.ds = ds
        self.ds_sizes = np.array(ds_sizes)
        self.dictionary = dictionary
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.pad_go = pad_go

    def __getitem__(self, index):
        # Get the tokens.
        if isinstance(self.ds[index], dict):
            src_tokens = self.ds[index]['tokens']
        else:
            src_tokens = self.ds[index]
        src_tokens = list(src_tokens)
        if self.pad_go:
            src_tokens.insert(0, self.dictionary.go())
        # Get masked src_tokens and label_tokens.
        src_tokens, masked_label_tokens = self.mask_src_tokens(src_tokens)
        src_tokens = torch.LongTensor(src_tokens)
        masked_label_tokens = torch.LongTensor(masked_label_tokens)

        return {
            'id': index,
            'source': src_tokens,
            'target': masked_label_tokens
        }

    def mask_src_tokens(self, src_tokens, p=0.15):
        # Set default of the masked label tokens as <pad> tokens.
        masked_label_tokens = [self.dictionary.pad()] * len(src_tokens)

        # Get indexes excluding <go>/<cls> and <eos>.
        cand_indexes = []
        for i, token in enumerate(src_tokens):
            if token == self.dictionary.go() or token == self.dictionary.eos():
                continue
            cand_indexes.append(i)
        random.shuffle(cand_indexes)
        # Decide the number of tokens to be masked.
        num_to_predict = max(1, int(round(len(src_tokens) * p)))
        cand_indexes = cand_indexes[:num_to_predict]

        # Mask selected tokens.
        for index in cand_indexes:
            # 80% chance to mask the token with <mask>.
            if random.random() < 0.8:
                masked_token = self.dictionary.mask()
            else:
                # 10% chance to mask the token with itself (unchanged).
                if random.random() < 0.5:
                    masked_token = src_tokens[index]
                # 10% chance to mask the token with random token.
                else:
                    masked_token = random.randint(0, len(self.dictionary) - 1)
            masked_label_tokens[index] = src_tokens[index]
            src_tokens[index] = masked_token
        return src_tokens, masked_label_tokens

    def __len__(self):
        return len(self.ds)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        # print("samples inside collater: ", samples)
        return collate(
            samples,
            pad_idx=self.dictionary.pad(),
            eos_idx=self.dictionary.eos(),
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target)

    def get_dummy_batch(self,
                        num_tokens,
                        max_positions,
                        src_len=128,
                        tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = max(num_tokens // max(src_len, tgt_len), 1)
        # print("missing???? ", self.has_missing)
        return self.collater([{
            'id':
            i,
            'source':
            self.dictionary.dummy_sentence(src_len),
            'target':
            self.dictionary.dummy_sentence(tgt_len)
        } for i in range(bsz)])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.ds_sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return [self.ds_sizes[index]]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            indices = indices[np.argsort(
                self.ds_sizes[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def supports_prefetch(self):
        return getattr(self.ds, 'supports_prefetch', False)

    def prefetch(self, indices):
        self.ds.prefetch(indices)
