import torch
from torch import nn
import numpy as np

from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList
from models.utils import one_hot_to_index


class MeshedDecoderLayer(Module):
    def __init__(self, N_enc, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderLayer, self).__init__()
        self.N_enc = N_enc
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.fc_alpha = ModuleList([nn.Linear(d_model + d_model, d_model) for _ in range(N_enc)])

        self.init_weights()

    def init_weights(self):
        for fc_alpha in self.fc_alpha:
            nn.init.xavier_uniform_(fc_alpha.weight)
            nn.init.constant_(fc_alpha.bias, 0)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad

        enc_att = None
        for i in range(self.N_enc):
            enc_att_i = self.enc_att(self_att, enc_output[:, i], enc_output[:, i], mask_enc_att) * mask_pad
            alpha_i = torch.sigmoid(self.fc_alpha[i](torch.cat([self_att, enc_att_i], -1)))
            if enc_att is None:
                enc_att = enc_att_i * alpha_i
            else:
                enc_att += enc_att_i * alpha_i

        enc_att /= np.sqrt(self.N_enc)
        enc_att *= mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)
        enc_att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class MeshedDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, N_enc, padding_idx=0, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoder, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [MeshedDecoderLayer(N_enc, d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', None)
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output_list, mask_encoder):
        # input (b_s, seq_len)
        input = input[:, :self.max_len]
        b_s, seq_len = input.shape[:2]

        if input.dtype in [torch.long, torch.int]:
            input_index = input
        else:
            input_index = one_hot_to_index(input)

        mask_queries = (input_index != self.padding_idx).unsqueeze(-1).type(input.dtype)  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input_index == self.padding_idx).unsqueeze(1).unsqueeze(1).bool()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            if self.running_mask_self_attention is None:
                self.running_mask_self_attention = mask_self_attention
            else:
                self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention],
                                                             -1)
                mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        if input.dtype in [torch.long, torch.int]:
            out = self.word_emb(input)
        else:
            out = input @ self.word_emb.weight

        out = out + self.pos_emb(seq)
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output_list, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return out


class Decoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx=0, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=.1, self_att_module=None, enc_att_module=None, self_att_module_kwargs=None,
                 enc_att_module_kwargs=None):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                          enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                          enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', None)
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        # input (b_s, seq_len)
        input = input[:, :self.max_len]
        b_s, seq_len = input.shape[:2]

        if input.dtype in [torch.long, torch.int]:
            input_index = input
        else:
            input_index = one_hot_to_index(input)

        mask_queries = (input_index != self.padding_idx).unsqueeze(-1).type(input.dtype)  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input_index == self.padding_idx).unsqueeze(1).unsqueeze(1).bool()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            if self.running_mask_self_attention is None:
                self.running_mask_self_attention = mask_self_attention
            else:
                self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention],
                                                             -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        if input.dtype in [torch.long, torch.int]:
            out = self.word_emb(input)
        else:
            out = input @ self.word_emb.weight

        out = out + self.pos_emb(seq)
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return out
