# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.modules import (AdaptiveInput, AdaptiveSoftmax,
                             CharacterTokenEmbedder,
                             LearnedPositionalEmbedding, MultiheadAttention,
                             SinusoidalPositionalEmbedding)

from . import (BaseFairseqModel, FairseqEncoder, FairseqIncrementalDecoder,
               FairseqLanguageModel, FairseqModel, PredNet, register_model,
               register_model_architecture)


@register_model('bert')
class BertModel(BaseFairseqModel):
    """
    BERT model from paper arxiv.org/abs/1810.04805.
    
    This implementation only contains the first training task in BERT paper,
    the Masked Language Modeling.
    Two pointers for BERT implementations are:
    1) github.com/google-research/bert
    2) github.com/huggingface/pytorch-pretrained-BERT

    Args:
        encoder: BERT encoder, built with Transformer layers.
        pooler: BERT pooler, pooling the output of the BERT encoder, pooling in
                this case meaning get the output of the BERT encoder's first
                token corresponding to <cls>, only used in fine-tuning.
        pretrain_head: BERT pretrain head for MLM, only used in pretraining.
        

    The BERT model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.bert_parser
        :prog:
    """

    def __init__(self,
                 encoder,
                 pretrain_head,
                 pooler,
                 prop_predict=False,
                 prednet=None):
        super().__init__()
        self.encoder = encoder
        self.pretrain_head = pretrain_head
        self.pooler = pooler
        self.prop_predict = prop_predict
        self.prednet = prednet

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument(
            '--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument(
            '--attention-dropout',
            type=float,
            metavar='D',
            help='dropout probability for attention weights')
        parser.add_argument(
            '--relu-dropout',
            type=float,
            metavar='D',
            help='dropout probability after ReLU in FFN')
        parser.add_argument(
            '--encoder-embed-path',
            type=str,
            metavar='STR',
            help='path to pre-trained encoder embedding')
        parser.add_argument(
            '--encoder-embed-dim',
            type=int,
            metavar='N',
            help='encoder embedding dimension')
        parser.add_argument(
            '--encoder-ffn-embed-dim',
            type=int,
            metavar='N',
            help='encoder embedding dimension for FFN')
        parser.add_argument(
            '--encoder-layers',
            type=int,
            metavar='N',
            help='num encoder layers')
        parser.add_argument(
            '--encoder-attention-heads',
            type=int,
            metavar='N',
            help='num encoder attention heads')
        parser.add_argument(
            '--encoder-normalize-before',
            action='store_true',
            help='apply layernorm before each encoder block')
        parser.add_argument(
            '--encoder-learned-pos',
            action='store_true',
            help='use learned positional embeddings in the encoder')
        parser.add_argument(
            '--no-token-positional-embeddings',
            default=False,
            action='store_true',
            help=
            'if set, disables positional embeddings (outside self attention)')
        parser.add_argument(
            '--adaptive-softmax-cutoff',
            metavar='EXPR',
            help='comma separated list of adaptive softmax cutoff points. '
            'Must be used with adaptive_loss criterion'),
        parser.add_argument(
            '--adaptive-softmax-dropout',
            type=float,
            metavar='D',
            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument(
            '--prop-predict',
            default=False,
            action='store_true',
            help='if to use the the prediction net for property prediction')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        # if not hasattr(args, 'max_target_positions'):
        #     args.max_target_positions = 1024

        src_dict = task.source_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim,
                                               args.encoder_embed_path)
        # BERT Encoder.
        encoder = TransformerEncoder(
            args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        # BERT PretrainMLMHead.
        pretrain_head = BertMLMHead(args, encoder_embed_tokens.weight)
        if args.prop_predict:
            # BERT Pooler.
            pooler = BertPooler(args)
            # PredNet.
            pred_input_dim = args.encoder_embed_dim
            prednet = PredNet(
                input_dim=pred_input_dim,
                hidden_dim=args.pred_hidden_dim,
                act_func=args.pred_act,
                num_props=args.num_props,
                dropout=args.pred_dropout,
                cls_index=args.cls_index)
        else:
            pooler = None
            prednet = None

        return BertModel(encoder, pretrain_head, pooler, args.prop_predict,
                         prednet)

    def forward(self, src_tokens, src_lengths):
        encoder_out = self.encoder(src_tokens, src_lengths)
        if self.prop_predict:
            # Changed version. Output both encoder output and predicted output.
            x = self.pretrain_head(encoder_out['encoder_out'])
            # Fine-tuning stage.
            fp = self.pooler(encoder_out['encoder_out'])
            bsz, csz = fp.shape
            pred_out = self.prednet(fp)
            return {'pred': pred_out, 'encoder_out': x}
        else:
            # Pre-training stage.
            x = self.pretrain_head(encoder_out['encoder_out'])
            return {'pred': None, 'encoder_out': x}


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded
            (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions,
            embed_dim,
            self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        return {
            'encoder_out': x,  # B x T x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions,
                   self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(
                name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList(
            [LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """

        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = gelu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class BertMLMHead(nn.Module):
    """BERT MLM Head for pretraining.
    """

    def __init__(self, args, bert_model_embedding_weights):
        super(BertMLMHead, self).__init__()
        # Nonlinear Mapping.
        self.embed_dim = args.encoder_embed_dim
        self.dense = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm = LayerNorm(self.embed_dim)
        # Decoder using the token embedding weights.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(
            torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, x):
        # Non-linear transform.
        x = gelu(self.dense(x))
        x = self.layer_norm(x)
        # Decoder.
        x = self.decoder(x) + self.bias
        return x


class BertPooler(nn.Module):
    """BERT pooler.
    """

    def __init__(self, args):
        super(BertPooler, self).__init__()
        self.embed_dim = args.encoder_embed_dim
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        x = x[:, 0, :]
        x = self.tanh(self.dense(x))
        return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def gelu(x):
    """Implementation of the gelu activation function.
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def PositionalEmbedding(num_embeddings,
                        embedding_dim,
                        padding_idx,
                        left_pad,
                        learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1,
                                       embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad,
                                          num_embeddings + padding_idx + 1)
    return m


@register_model_architecture('bert', 'bert')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before',
                                            False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff',
                                           None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(
        args, 'share_decoder_input_output_embed', False)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)


@register_model_architecture('bert', 'bertlarge')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before',
                                            False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff',
                                           None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)


@register_model_architecture('bert', 'bertsmall')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before',
                                            False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff',
                                           None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
