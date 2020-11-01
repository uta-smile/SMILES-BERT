# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq import utils

from . import FairseqDataset, data_utils


def collate(samples,
            pad_idx,
            eos_idx,
            left_pad_source=True,
            left_pad_target=False,
            input_feeding=True,
            has_prop=False,
            has_missing=False):
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

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    if has_prop:
        prop = torch.stack([s['prop'] for s in samples])
        prop = prop.index_select(0, sort_order)
    else:
        prop = None

    if has_missing:
        miss = torch.stack([s['miss'] for s in samples])
        miss = miss.index_select(0, sort_order)
    else:
        miss = None
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'prop': prop,
        'miss': miss
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class SmilePredictionDataset(FairseqDataset):
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
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(self,
                 ds,
                 ds_sizes,
                 dictionary,
                 left_pad_source=True,
                 left_pad_target=False,
                 max_source_positions=256,
                 max_target_positions=256,
                 shuffle=True,
                 reverse_input=False,
                 pad_go=True,
                 input_feeding=True):
        self.ds = ds
        self.ds_sizes = np.array(ds_sizes)
        self.dictionary = dictionary
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.reverse_input = reverse_input
        self.pad_go = pad_go
        self.num_props = getattr(ds, 'num_props', 0)
        self.has_prop = self.num_props > 0
        self.has_missing = getattr(ds, 'has_missing', False)

    def __getitem__(self, index):
        if isinstance(self.ds[index], dict):
            tgt_item = self.ds[index]['tokens']
        else:
            tgt_item = self.ds[index]
        src_list = list(reversed(
            tgt_item[:-1])) if self.reverse_input else list(tgt_item[:-1])
        if self.pad_go:
            src_list.insert(0, self.dictionary.go())
            tgt_list = list(tgt_item)
            tgt_list.insert(0, self.dictionary.go())
            tgt_item = torch.LongTensor(tgt_list)
        src_item = torch.LongTensor(src_list)

        result_dict = {'id': index, 'source': src_item, 'target': tgt_item}

        if self.has_prop:
            result_dict['prop'] = self.ds[index]['prop']
            if self.has_missing:
                result_dict['miss'] = self.ds[index]['miss']
        return result_dict

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
        return collate(
            samples,
            pad_idx=self.dictionary.pad(),
            eos_idx=self.dictionary.eos(),
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            has_prop=self.has_prop,
            has_missing=self.has_missing)

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
            self.dictionary.dummy_sentence(tgt_len),
            'prop':
            torch.empty(self.num_props).random_(2) if self.has_prop else None,
            'miss':
            torch.empty(self.num_props, dtype=torch.int).random_(2)
            if self.has_missing else None
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
