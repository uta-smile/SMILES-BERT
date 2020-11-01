# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn.functional as F
from torch import nn

from fairseq import utils

from . import FairseqCriterion, register_criterion
from .criterion_utils import em_accuracy


@register_criterion('seq3seq')
class Seq3seqCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.alpha = args.sup_loss_alpha
        self.to_recover = 1.0 if args.to_recover else 0.0
        self.num_props = args.num_props
        self.cls_index = [] if -1 in args.cls_index else args.cls_index
        self.prop_weight = args.prop_weight
        self.has_missing = args.has_missing

        self.cls_criterion = nn.CrossEntropyLoss(reduction='none')
        self.reg_criterion = nn.MSELoss(reduction='none')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # print(sample['net_input'])
        net_output = model(**sample['net_input'])
        assert len(net_output) == 2, 'Length of model output should be 2.'
        if isinstance(net_output, dict):
            assert 'encoder_out' in net_output, 'encoder_out should be in net_output.'
            assert 'pred' in net_output, 'pred should be in net_output'
            net_pred = net_output['pred']
            net_output = net_output['encoder_out']
        else:
            net_output, net_pred = net_output

        # Unsupervised criterion.
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        # print('*****', target.shape)
        # print('*****', lprobs.shape)
        if lprobs.shape[1] != target.shape[1]:
            assert lprobs.shape[1] + 1 == target.shape[1], "The target has EOS."
            target = target[:, :-1]
        _, pred = lprobs.max(2)
        # print(lprobs.view(-1, lprobs.size(-1)).shape)
        # print(target.contiguous().view(-1).shape)
        unsup_loss = F.nll_loss(
            lprobs.view(-1, lprobs.size(-1)),
            target.contiguous().view(-1),
            size_average=False,
            ignore_index=self.padding_idx,
            reduce=reduce)
        # em_acc
        em_acc = em_accuracy(pred, target, self.padding_idx)

        # Supervised criterion.
        # Check if the weight is None.
        if not self.prop_weight:
            if len(self.prop_weight) == self.num_props:
                prop_weight = self.prop_weight
            else:
                prop_weight = [1.0] * self.num_props
        else:
            prop_weight = [1.0] * self.num_props
        prop = sample['prop']
        if self.has_missing:
            miss = sample['miss']
            assert miss is not None, "Miss should not be None type."
        else:
            miss = torch.ones_like(prop, dtype=torch.int)
        sample_size_per_prop = miss.sum(0).tolist()
        miss = miss.float()
        sup_loss = 0
        sup_loss_details = {}
        for i in range(self.num_props):
            if i in self.cls_index:
                current_loss = self.cls_criterion(net_pred[i],
                                                  prop[:, i].long())
                if sample_size_per_prop[i] == 0:
                    current_loss = current_loss.sum() * 0
                else:
                    current_loss = (miss[:, i] * current_loss
                                   ).sum() / sample_size_per_prop[i]
                sup_loss_details['%d_cls' % i] = current_loss
                # Add accuracy.
                _, pred_cls = net_pred[i].max(1)
                acc = (
                    (pred_cls == prop[:, i].long()).float() * miss[:, i]).sum()
                sup_loss_details['%d_acc' % i] = acc / sample_size_per_prop[i]
            else:
                current_loss = self.reg_criterion(
                    torch.squeeze(net_pred[i]), prop[:, i])
                if sample_size_per_prop[i] == 0:
                    current_loss = current_loss.sum() * 0
                else:
                    current_loss = (miss[:, i] * current_loss
                                   ).sum() / sample_size_per_prop[i]
                sup_loss_details['%d_reg' % i] = current_loss
            sup_loss_details['%d_num' % i] = sample_size_per_prop[i]
            sup_loss += prop_weight[i] * current_loss
        loss = self.to_recover * unsup_loss + self.alpha * sup_loss
        # loss = sup_loss
        sample_size = sample['target'].size(
            0) if self.args.sentence_avg else sample['ntokens']
        # print("*****", sup_loss)
        # print(hjey)
        logging_output = {
            'unsup_loss': utils.item(unsup_loss) if reduce else unsup_loss.data,
            'sup_loss': utils.item(sup_loss) if reduce else sup_loss.data,
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'em_ac': em_acc
        }

        for i in sup_loss_details:
            if i not in logging_output:
                logging_output[i] = utils.item(
                    sup_loss_details[i]) if reduce else sup_loss_details[i].data
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs, args):
        """Aggregate logging outputs from data parallel training."""
        unsup_loss_sum = sum(
            log.get('unsup_loss', 0) for log in logging_outputs)
        sup_loss_sum = sum(log.get('sup_loss', 0) for log in logging_outputs)
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        em_acc = sum(log.get('em_ac', 0) for log in logging_outputs)
        agg_output = {
            'unsup_loss': unsup_loss_sum / sample_size / math.log(2),
            'sup_loss': sup_loss_sum / math.log(2),
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'em_ac': em_acc
        }

        for i in range(args.num_props):
            loss_log_key = '%d_cls' % i if i in args.cls_index else '%d_reg' % i
            sample_num_log_key = '%d_num' % i
            sample_num_sum = sum(
                log.get(sample_num_log_key, 0) for log in logging_outputs)
            if sample_num_sum == 0:
                agg_output[loss_log_key] = 0.0
            else:
                agg_output[loss_log_key] = sum(
                    log.get(loss_log_key, 0) * log.get('%d_num' % i, 0)
                    for log in logging_outputs) / sample_num_sum
            agg_output[sample_num_log_key] = sample_num_sum
            if i in args.cls_index:
                if sample_num_sum == 0:
                    agg_output['%d_acc' % i] = 0.0
                else:
                    agg_output['%d_acc' % i] = sum(
                        log.get('%d_acc' % i, 0) * log.get('%d_num' % i, 0)
                        for log in logging_outputs) / sample_num_sum
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output

    def _aggregate_logging_outputs(self, logging_outputs, args):
        """An instance method version of :func:`aggregate_logging_outputs`.

        This can be overridden if needed, but please be careful not to rely
        on shared state when aggregating logging outputs otherwise you may
        get incorrect results.
        """
        return self.__class__.aggregate_logging_outputs(logging_outputs, args)
