# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

from fairseq import options, utils
from fairseq.data import (BertDataset, ConcatDataset, Dictionary,
                          GeneralSmileDictionary, IndexedCachedDataset,
                          IndexedDataset, IndexedRawTextDataset,
                          IndexedSmilePropertyDataset, LanguagePairDataset,
                          SmileDictionary, SmilePredictionDataset, data_utils)

from . import FairseqTask, register_task


@register_task('bert')
class BertTask(FairseqTask):
    """
    BertTask for BERT pre-training.

    Args:
        smile_dict (Dictionary): dictionary for molecule SMILEs.

    .. note::

        The predict task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument(
            'data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument(
            '--lazy-load', action='store_true', help='load the dataset lazily')
        parser.add_argument(
            '--left-pad-source',
            default='False',
            type=str,
            metavar='BOOL',
            help='pad the source on the left')
        parser.add_argument(
            '--left-pad-target',
            default='False',
            type=str,
            metavar='BOOL',
            help='pad the target on the left')
        parser.add_argument(
            '--max-source-positions',
            default=1024,
            type=int,
            metavar='N',
            help='max number of tokens in the source sequence')
        parser.add_argument(
            '--max-target-positions',
            default=1024,
            type=int,
            metavar='N',
            help='max number of tokens in the target sequence')
        parser.add_argument(
            '--header',
            default='True',
            type=str,
            metavar='BOOL',
            help='if data file contains header')
        parser.add_argument(
            '--num-props',
            default=0,
            type=int,
            help='number of smile properties')
        parser.add_argument(
            '--separator', default=',', type=str, help='data file separator')
        parser.add_argument(
            '--prop-weight',
            default='1.0',
            metavar='P0,P1,...,PN',
            help='properties weight in loss')
        parser.add_argument(
            '--cls-index',
            default='-1',
            metavar='P0,P1,...,PN',
            help='index for classification')
        parser.add_argument(
            '--has-missing',
            action='store_true',
            help='if data file contains missing property')
        parser.add_argument(
            '--missing-symbol',
            default='NaN',
            type=str,
            help='missing property symbol')
        parser.add_argument(
            '--sup-loss-alpha',
            default=1.0,
            type=float,
            help='supervised loss weight alpha')
        parser.add_argument(
            '--to-recover',
            action='store_true',
            help='if use the unsupervised loss')
        parser.add_argument(
            '--data-bin',
            action='store_true',
            help='if load binary processed data')
        # Settings for PredNet.
        parser.add_argument(
            '--pred-hidden-dim',
            default=512,
            type=int,
            metavar='N',
            help='Hidden layer size in PredNet.')
        parser.add_argument(
            '--pred-dropout',
            default=0.5,
            type=float,
            metavar='D',
            help='Dropout rate in PredNet hidden layer.')
        parser.add_argument(
            '--pred-act',
            default='LeakyReLU',
            type=str,
            metavar='STR',
            help='Activation function for PredNet hidden layer.')
        # Additional settings for Dataset.
        parser.add_argument(
            '--pad-go',
            default='True',
            type=str,
            metavar='BOOL',
            help='If pad the <go> on the left for the input sequence.')
        parser.add_argument(
            '--reverse-input',
            default='False',
            type=str,
            metavar='BOOL',
            help='If reverse the input sequence.')
        parser.add_argument(
            '--bert-pretrain',
            default='False',
            type=str,
            metavar='BOOL',
            help='If bert pretrain stage.')
        # to support new dictionary and tokenizer.
        parser.add_argument(
            '--smile-dic-type',
            default='short',
            type=str,
            metavar='STR',
            help='Which SMILEs dictionary type to use.')
        # fmt: on

    @staticmethod
    def load_pretrained_model(path, arg_overrides=None):
        model = utils.load_checkpoint_to_cpu(path)
        args = model['args']
        state_dict = model['model']
        args = utils.override_model_args(args, arg_overrides)
        if args.smile_dic_type == 'short':
            dictionary = SmileDictionary.load()
        else:
            dictionary = GeneralSmileDictionary.load()

        task = SmilePropertyPredictionTask(args, dictionary)
        model = task.build_model(args)
        model.upgrade_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dic = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        args.header = options.eval_bool(args.header)
        args.pad_go = options.eval_bool(args.pad_go)
        args.reverse_input = options.eval_bool(args.reverse_input)
        args.bert_pretrain = options.eval_bool(args.bert_pretrain)

        assert not args.left_pad_source, "Source should be right padded."
        assert not args.left_pad_target, "Target should be right padded."
        assert not args.reverse_input, "Source should not be reverted."
        assert args.pad_go, "<GO> or <CLS> token should be padded in the begging."

        # load dictionaries
        if args.smile_dic_type == 'short':
            dictionary = SmileDictionary.load()
        else:
            dictionary = GeneralSmileDictionary.load()
        print('| [{}] dictionary: {} types'.format("SMILEs", len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, shuffle=True, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        print("| Loading split: ", split)

        data_paths = self.args.data
        data_path = data_paths[0]
        split_path = os.path.join(data_path, split)
        # if not os.path.exists(split_path):
        #     raise FileNotFoundError('Dataset not found: {}'.format(split_path))

        if self.args.data_bin:
            ds = IndexedDataset(split_path, fix_lua_indexing=True)
        else:
            ds = IndexedSmilePropertyDataset(
                split_path,
                self.dic,
                header=self.args.header,
                num_props=self.args.num_props,
                has_missing=self.args.has_missing,
                tokenizer=self.args.smile_dic_type)

        if self.args.bert_pretrain:
            self.datasets[split] = BertDataset(
                ds,
                ds.sizes,
                self.dic,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                shuffle=shuffle,
                pad_go=self.args.pad_go)

        else:
            self.datasets[split] = SmilePredictionDataset(
                ds,
                ds.sizes,
                self.dic,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                shuffle=shuffle,
                reverse_input=self.args.reverse_input,
                pad_go=self.args.pad_go)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dic

    @property
    def target_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dic

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        if self.args.prop_predict:
            return criterion._aggregate_logging_outputs(logging_outputs,
                                                        self.args)
        else:
            return criterion._aggregate_logging_outputs(logging_outputs)
