from torch.nn import functional as F
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class Encoder(nn.Module):
    def __init__(self, N, max_len, d_in, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None,
                 with_pe=False, with_mesh=False):
        super(Encoder, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, self.d_in, 0), freeze=True)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.with_pe = with_pe
        self.with_mesh = with_mesh

    def forward(self, input):
        # input (b_s, seq_len, d_in)
        b_s, seq_len = input.shape[:2]
        seq = torch.arange(1, seq_len + 1, device=input.device).view(1, -1).expand(b_s, -1)  # (b_s, seq_len)

        out = input
        if self.with_pe:
            out = out + self.pos_emb(seq)
        out = F.relu(self.fc(out))
        out = self.dropout(out)
        out = self.layer_norm(out)
        outs = list()
        for l in self.layers:
            out = l(out, out, out)
            if self.with_mesh:
                outs.append(out.unsqueeze(1))

        if self.with_mesh:
            outs = torch.cat(outs, 1)
            return outs, None
        return out, None

