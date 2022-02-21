import copy
from pathlib import Path

import torch
from torch import Tensor
from torch import nn

from data.field import TextField
from models.beam_search import *
from models.containers import ModuleList, Module
from utils import TensorOrSequence
from . import Encoder, Decoder, ScaledDotProductAttentionMemory, MeshedDecoder


class Captioner(Module):
    def __init__(self, args, text_field: TextField):
        super(Captioner, self).__init__()

        self.encoder = Encoder(args.N_enc, 500, args.image_dim, d_model=args.d_model, d_ff=args.d_ff, h=args.head,
                               attention_module=ScaledDotProductAttentionMemory,
                               attention_module_kwargs={'m': args.m},
                               with_pe=args.with_pe, with_mesh=not args.disable_mesh)
        if args.disable_mesh:
            self.decoder = Decoder(text_field._tokenizer.vocab_size, 40, args.N_dec, d_model=args.d_model,
                                   d_ff=args.d_ff, h=args.head)
        else:
            self.decoder = MeshedDecoder(text_field._tokenizer.vocab_size, 40, args.N_dec, args.N_enc,
                                         d_model=args.d_model, d_ff=args.d_ff, h=args.head)
        self.bos_idx = text_field._tokenizer.bos_idx
        self.eos_idx = text_field._tokenizer.eos_idx
        self.vocab_size = text_field._tokenizer.vocab_size
        self.max_generation_length = self.decoder.max_len

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        self.decoder.train(mode)

    def init_weights(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq):
        enc_output, mask_enc = self.encoder(images)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def step(self, t: int, prev_output: Tensor, visual: Tensor) -> Tensor:
        if t == 0:
            self.enc_output, self.mask_enc = self.encoder(visual)
            input = visual.data.new_full((visual.shape[0], 1), self.bos_idx, dtype=torch.long)
        else:
            input = prev_output
        logits = self.decoder(input, self.enc_output, self.mask_enc)
        return logits

    def beam_search(self, visual: TensorOrSequence, beam_size: int, out_size=1,
                    return_logits=False, **kwargs):
        bs = BeamSearch(self, self.max_generation_length, self.eos_idx, beam_size)
        return bs.apply(visual, out_size, return_logits, **kwargs)


class CaptionerEnsemble(Captioner):
    def __init__(self, model: Captioner, args, text_field, weight_files, weight_folder=None):
        super(CaptionerEnsemble, self).__init__(args, text_field)
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for model_i, weight_file_i in zip(self.models, weight_files):
            if Path(weight_file_i).is_absolute():
                fname = Path(weight_file_i)
            else:
                fname = Path(weight_folder).joinpath(weight_file_i)
            state_dict_i = torch.load(fname)['state_dict_t']
            model_i.load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual):
        out_ensemble = []
        for model_i in self.models:
            out_i = model_i.step(t, prev_output, visual)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
