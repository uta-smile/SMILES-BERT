#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import os
import shutil
from collections import Counter
from itertools import zip_longest
from multiprocessing import Pool

from fairseq.data import dictionary, indexed_dataset
from fairseq.data.dictionary import GeneralSmileDictionary, SmileDictionary
from fairseq.tokenizer import GeneralSmileTokenizer, SmileTokenizer, Tokenizer


def get_parser():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument(
        '--data',
        metavar="DIR",
        default="dummy/zinc",
        help='path to data directory')
    parser.add_argument(
        "--destdir", metavar="DIR", default="data-bin", help="destination dir")
    parser.add_argument(
        "--workers",
        metavar="N",
        default=1,
        type=int,
        help="number of parallel workers")
    # to support new dictionary and tokenizer.
    parser.add_argument(
        '--smile-dic-type',
        default='short',
        type=str,
        metavar='STR',
        help='Which SMILEs dictionary type to use.')
    # fmt: on
    return parser


def main(args):
    print(args)
    os.makedirs(args.destdir, exist_ok=True)

    if args.smile_dic_type == 'short':
        dic = SmileDictionary.load()
        args.tokenizer = SmileTokenizer
    else:
        dic = GeneralSmileDictionary.load()
        args.tokenizer = GeneralSmileTokenizer
    print("| SMILEs Dictionary: {} types".format(len(dic) - 1))

    def make_binary_dataset(input_file, output_prefix, dic, num_workers):
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        offsets = args.tokenizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        prefix,
                        dic,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = indexed_dataset.IndexedDatasetBuilder(
            output_file(args, output_prefix, "bin"))
        merge_result(
            args.tokenizer.binarize(
                input_file,
                dic,
                lambda t: ds.add_item(t),
                offset=0,
                end=offsets[1]))
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = output_file(args, prefix, '')
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(output_file(args, output_prefix, 'idx'))

        print("| {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
            input_file, n_seq_tok[0], n_seq_tok[1],
            100 * sum(replaced.values()) / n_seq_tok[1], dic.unk_word))

    def make_all():

        def source_file(data_path, prefix):
            source_file = os.path.join(data_path, prefix)
            return source_file

        train_file = source_file(args.data, 'train')
        valid_file = source_file(args.data, 'valid')
        test_file = source_file(args.data, 'test')
        make_binary_dataset(train_file, 'train', dic, num_workers=args.workers)
        make_binary_dataset(valid_file, 'valid', dic, num_workers=args.workers)
        make_binary_dataset(test_file, 'test', dic, num_workers=args.workers)

    make_all()

    print("| Wrote preprocessed data to {}".format(args.destdir))


def output_file(args, output_prefix, postfix='bin'):
    if postfix:
        postfix = '.' + postfix
    return os.path.join(args.destdir, output_prefix + postfix)


def binarize(args, input_file, output_prefix, dict, offset, end):
    ds = indexed_dataset.IndexedDatasetBuilder(
        output_file(args, output_prefix, 'bin'))

    def consumer(tensor):
        ds.add_item(tensor)

    res = args.tokenizer.binarize(
        input_file, dict, consumer, offset=offset, end=end)
    ds.finalize(output_file(args, output_prefix, 'idx'))
    return res


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
