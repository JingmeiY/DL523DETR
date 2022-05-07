"""
DETR Transformer class.
"reimplement code of (c) Facebook, Inc. and its affiliates"
from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # self_attn use to calculate the attention between object queries
        # multihead use to calculate the attention between object queries and output from encoder
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation) # choose activation function from [relu, gelu, glu]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is None:
            output = tensor
        else:
            output = tensor + pos
        return output


    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """
        normalization first
        normalize input before self attention
        """
        tgt2 = self.norm1(tgt)

        # self attention between object queries in decoder
        q = self.with_pos_embed(tgt2, query_pos)
        k = self.with_pos_embed(tgt2, query_pos)
        tgt2, tgt2_attn_weight = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                                key_padding_mask=tgt_key_padding_mask)  # calculate the self attention
        tgt2 = self.dropout1(tgt2) # dropout layer
        tgt = tgt + tgt2  # residual connection
        tgt2 = self.norm2(tgt) # normalization after attention

        # attention calculation between weighted object queries from previous self attention ouput and output from encoder
        tgt2, tgt2_attn_weight = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                                     key=self.with_pos_embed(memory, pos),
                                                     value=memory, attn_mask=memory_mask,
                                                     key_padding_mask=memory_key_padding_mask)
        tgt2 = self.dropout2(tgt2) # dropout layer

        tgt = tgt + tgt2  # residual connection

        tgt2 = self.norm3(tgt)  # normalization after attention

        # feed forward fully connection layer
        tgt2 = self.linear1(tgt2)
        tgt2 = self.activation(tgt2)
        tgt2 = self.dropout(tgt2) # dropout layer in fully connection
        tgt2 = self.linear2(tgt2)

        tgt2 = self.dropout3(tgt2) # dropout layer

        tgt = tgt + tgt2 # residual connection

        return tgt

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        normalization after attention
        normalize output of attention
        """

        q =  self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(tgt, query_pos)
        # self attention between object queries in decoder
        tgt2, tgt2_attn_weight = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                                key_padding_mask=tgt_key_padding_mask)  # calculate the self attention
        tgt2 = self.dropout1(tgt2)  # dropout layer
        tgt = tgt + tgt2  # residual connection
        tgt = self.norm1(tgt)  # normalization after attention

        # attention calculation between weighted object queries from previous self attention ouput and output from encoder
        tgt2, tgt2_attn_weight = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                                     key=self.with_pos_embed(memory, pos),
                                                     value=memory, attn_mask=memory_mask,
                                                     key_padding_mask=memory_key_padding_mask)
        tgt2 = self.dropout2(tgt2)  # dropout layer

        tgt = tgt + tgt2  # residual connection

        tgt = self.norm2(tgt)  # normalization after attention

        # feed forward fully connection layer
        tgt2 = self.linear1(tgt)
        tgt2 = self.activation(tgt2)
        tgt2 = self.dropout(tgt2) # dropout layer in fully connection
        tgt2 = self.linear2(tgt2)


        tgt2 = self.dropout3(tgt2)  # dropout layer
        tgt = tgt + tgt2  # residual connection
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            tgt = self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        else:
            tgt = self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return tgt


class TransformerDecoder(nn.Module):
    """"
    decoder_layer: decoder block: includes a self attention between object queries, a multi-head attention between object queries and output from
    encoder, a fully connect layer
    num_layers: number of decoder blocks

    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers) #  module list with repeated encoder block
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):


        intermediate = []# empty list to store the intermediate result

        # input of next block is the output from the previous block
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate_result = self.norm(output)
                intermediate.append(intermediate_result)

        if self.norm is not None:
            output = self.norm(output) # normalization of the output
            if self.return_intermediate:
                intermediate = intermediate[:-1]
                intermediate.append(output)

        if self.return_intermediate:
            decoder_out = torch.stack(intermediate)
        else:
            decoder_out = output.unsqueeze(0)
        return decoder_out


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.layers =nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        

    def forward(self, src,mask: Optional[Tensor] = None,src_key_padding_mask: Optional[Tensor] = None,pos: Optional[Tensor] = None):

        Encoder_out=src
        for layer in self.layers:
            Encoder_out = layer(src, src_mask=mask,src_key_padding_mask=src_key_padding_mask, pos=pos)


        if self.norm is not None:
            output = self.norm(Encoder_out)
            return output
        else:
            return Encoder_out

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.model=d_model
        self.dimf=dim_feedforward
        self.normalize_before = normalize_before
        
        self.linear1 = nn.Linear(self.model, self.dimf)
        self.linear2 = nn.Linear(self.dimf, self.model)

        self.norm1 = nn.LayerNorm(self.model)
        self.norm2 = nn.LayerNorm(self.model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(self.model, nhead, dropout=dropout)
        self.activation = _get_activation_fn(activation)
        

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is None:
            return tensor
        else:
            return tensor + pos

    def forward_post(self,src,src_mask: Optional[Tensor] = None,src_key_padding_mask: Optional[Tensor] = None,pos: Optional[Tensor] = None):
        q = self.with_pos_embed(src, pos)
        k = self.with_pos_embed(src, pos)
        atten = self.self_attn(q, k, value=src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(atten)
        linear_2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm1(src)))))
        src = self.norm1(src) + self.dropout2(linear_2)
        return self.norm2(src)

    def forward_pre(self, src,src_mask: Optional[Tensor] = None,src_key_padding_mask: Optional[Tensor] = None,pos: Optional[Tensor] = None):
        q = self.with_pos_embed(self.norm1(src), pos)
        k = self.with_pos_embed(self.norm1(src), pos)
        src2 = self.self_attn(q, k, value=self.norm1(src), attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        return src + self.dropout2(src2)

    def forward(self, src,src_mask: Optional[Tensor] = None,src_key_padding_mask: Optional[Tensor] = None,pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        else:
            return self.forward_post(src, src_mask, src_key_padding_mask, pos)
            
class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,activation="relu", normalize_before=False,return_intermediate_dec=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        if normalize_before:
            encoder_norm = nn.LayerNorm(d_model)
        else:
            encoder_norm =None
        decoder_norm = nn.LayerNorm(d_model)
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,dropout, activation, normalize_before)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,dropout, activation, normalize_before)
        
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        
        mask = mask.flatten(1)
        batch, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, batch, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(batch, c, h, w)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

