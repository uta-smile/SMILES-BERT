# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import re
from collections import Counter
from multiprocessing import Pool

import torch

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Tokenizer:

    @staticmethod
    def add_file_to_dictionary_single_worker(filename,
                                             tokenize,
                                             eos_word,
                                             worker_id=0,
                                             num_workers=1):
        counter = Counter()
        with open(filename, 'r') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):

        def merge_result(counter):
            for w, c in counter.items():
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        Tokenizer.add_file_to_dictionary_single_worker,
                        (filename, tokenize, dict.eos_word, worker_id,
                         num_workers)))
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                Tokenizer.add_file_to_dictionary_single_worker(
                    filename, tokenize, dict.eos_word))

    @staticmethod
    def binarize(
            filename,
            dict,
            consumer,
            tokenize=tokenize_line,
            append_eos=True,
            reverse_order=False,
            offset=0,
            end=-1,
    ):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = Tokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {
            'nseq': nseq,
            'nunk': sum(replaced.values()),
            'ntok': ntok,
            'replaced': replaced
        }

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, 'r') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets

    @staticmethod
    def tokenize(line,
                 dict,
                 tokenize=tokenize_line,
                 add_if_not_exist=True,
                 consumer=None,
                 append_eos=True,
                 reverse_order=False):
        words = tokenize(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dict.eos_index
        return ids


LEN_CHEM_ELEMENT = ["Cl", "Br", "10", "Ru"]


def smile_tokenize(smile):
    smile = smile.strip()
    idx = 0
    tokens = []
    while idx < len(smile):
        if idx < len(smile) - 1 and smile[idx:idx + 2] in LEN_CHEM_ELEMENT:
            token = smile[idx:idx + 2]
        else:
            token = smile[idx]
        idx += len(token)
        tokens.append(token)
    return tokens


class SmileTokenizer(Tokenizer):
    """Tokenizer for molecule SMILEs.
    """

    @staticmethod
    def tokenize(line,
                 dictionary,
                 tokenize=smile_tokenize,
                 add_if_not_exist=False,
                 consumer=None,
                 append_eos=True,
                 reversed_order=False):
        words = tokenize(line)
        if reversed_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dictionary.add_symbol(word)
            else:
                idx = dictionary.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dictionary.eos_index
        return ids

    @staticmethod
    def binarize(filename,
                 dict,
                 consumer,
                 tokenize=smile_tokenize,
                 append_eos=True,
                 reverse_order=False,
                 offset=0,
                 end=-1):
        return super(SmileTokenizer, SmileTokenizer).binarize(
            filename, dict, consumer, smile_tokenize, append_eos, reverse_order,
            offset, end)


def general_smile_tokenize(smile):
    """
    Tokenize a SMILES molecule or reaction
    """
    smile = smile.strip()
    import re
    pattern = "(\[[^\]]+]|Ru?|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smile)]
    # print('smile: ', smile)
    # print('tokens: ', ''.join(tokens))
    assert smile == ''.join(tokens)
    # return ' '.join(tokens)
    return tokens


class GeneralSmileTokenizer(Tokenizer):
    """
    """

    @staticmethod
    def tokenize(line,
                 dictionary,
                 tokenize=general_smile_tokenize,
                 add_if_not_exist=False,
                 consumer=None,
                 append_eos=True,
                 reversed_order=False):
        words = tokenize(line)
        if reversed_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dictionary.add_symbol(word)
            else:
                idx = dictionary.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dictionary.eos_index
        return ids

    @staticmethod
    def binarize(filename,
                 dict,
                 consumer,
                 tokenize=general_smile_tokenize,
                 append_eos=True,
                 reverse_order=False,
                 offset=0,
                 end=-1):
        return super(GeneralSmileTokenizer, GeneralSmileTokenizer).binarize(
            filename, dict, consumer, tokenize, append_eos, reverse_order,
            offset, end)
