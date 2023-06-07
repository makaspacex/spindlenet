import math

import torch
import torch.nn
from torch.nn import functional as trnf

from neko_2021_mjt.modulars.neko_inflater import NekoInflater
from neko_sdk.torchtools.seqkit import length_to_mask


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# This version needs no parameter
class NekoV2sBasic(torch.nn.Module):
    def __init__(self):
        super(NekoV2sBasic, self).__init__()
        pass

    # compacted sembs correspond to the visual prototypes, which inevitably contains
    # a lot of unknown symbols---be it visually unclear or simply left to be by the sampler.
    # yet, these prediction will work as skip-gram data (wwww)
    # Note in this version we do not try to change the length--- though it would be possible.
    # so we will not give the differentiable length prediction....
    def forward(self, logits_raw, lengths, compact_sembs, maxT):
        prob = trnf.softmax(logits_raw, -1)
        embs = prob.matmul(compact_sembs)
        # well, in mk8 we will prevent empty prediction to keep slicing sane.
        chunks = torch.split(embs, lengths)
        tlen = torch.tensor(lengths, device=logits_raw.device)
        masks = length_to_mask(tlen, maxT)
        sbatch = torch.nn.utils.rnn.pad_sequence(chunks)
        return sbatch, masks, tlen


from torch.nn import TransformerEncoder, Transformer
from neko_sdk.seq2seq.neko_fixed_torch_transformer import neko_TransformerEncoderLayer, neko_TransformerDecoderLayer


class NekoCtxBasic(torch.nn.Module):
    def __init__(self, feat_ch, numh=8, num_l=4):
        super(NekoCtxBasic, self).__init__()
        encoder_layer = neko_TransformerEncoderLayer(d_model=feat_ch, nhead=numh)
        # welp let's abuse the setup here.
        self.core = TransformerEncoder(encoder_layer, num_layers=num_l)
        pass

    def forward(self, sbatch, smask):
        cf = self.core(sbatch, src_key_padding_mask=smask)
        return cf


class NekoCtxEncdec(torch.nn.Module):
    def __init__(self, feat_ch, numh=8, num_l=4):
        super(NekoCtxEncdec, self).__init__()
        self.core = Transformer(d_model=feat_ch, nhead=numh, custom_encoder=neko_TransformerEncoderLayer,
                                custom_decoder=neko_TransformerDecoderLayer)
        pass

    def forward(self, sbatch, smask):
        cf = self.core(sbatch, src_key_padding_mask=smask)
        return cf


class NekoBasicCtxModule(torch.nn.Module):
    def __init__(self, feat_ch, nhead=8, nlay=4):
        super(NekoBasicCtxModule, self).__init__()
        self.indexmod = NekoV2sBasic()
        self.ctxmod = NekoCtxBasic(feat_ch, nhead, nlay)
        self.inflator = NekoInflater()

    def forward(self, logits_raw, lengths, compact_sembs, semb, maxT):
        sbatch, masks, tlen = self.indexmod(logits_raw, lengths, compact_sembs, maxT)
        cf = self.ctxmod(sbatch, ~masks)
        femb, _ = self.inflator.inflate(cf, tlen)
        logits = torch.mm(femb, semb.T)
        clogits = torch.mm(femb, compact_sembs.T)
        return logits, clogits
