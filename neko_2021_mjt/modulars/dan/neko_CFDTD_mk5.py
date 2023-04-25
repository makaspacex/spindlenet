import torch
from torch import nn

'''
Decoupled Text Decoder
'''


# these dtds does not decode prediction according to prev timestamp classifications.
# this is to get less bloating code. This starts from mk5.
# to support conventional APIs, go for GPDTDs

# 这个类的名字被写死在了pth文件中

class neko_os_CFDTD_mk5(nn.Module):

    def __init__(self):
        super(neko_os_CFDTD_mk5, self).__init__()
        self.setup_modules()
        self.baseline = 0

    def setup_modules(self, dropout=0.3):
        self.drop = dropout
        return

    def loop(self, C, nsteps, nB):
        out_emb = torch.zeros(nsteps, nB, C.shape[-1]).type_as(C.data)
        hidden = C
        out_emb[:nsteps, :, :] = hidden[:nsteps, :, :]
        return out_emb

    def getC(self, feature, A, nB, nC, nH, nW, nT):
        C = feature.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
        C = C.view(nB, nT, nC, -1).sum(3).transpose(1, 0)
        return C

    def sample(self, feature, A):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]
        # Normalize
        # OOF! is this the cause for the bleeding and performance impact?????
        A = A / (A.view(nB, nT, -1).sum(2).view(nB, nT, 1, 1) + 0.0001)
        # weighted sum
        C = self.getC(feature, A, nB, nC, nH, nW, nT)
        return A, C
        pass

    # we may need a forward_time_stamp here or may be insert a call back on the classifier. Let's see.
    def forward(self, feature, A, text_length):
        nB, nC, nH, nW = feature.size()
        A, C = self.sample(feature, A)
        if (self.training and text_length is not None):
            nsteps = int(text_length.max())
        else:
            nsteps = A.size()[1]
        out_emb = self.loop(C, nsteps, nB)
        # out_emb= trnf.dropout(self.loop(C,nsteps, nB),self.drop,self.training)
        return out_emb


class NekoOsCfdtdMk6(neko_os_CFDTD_mk5):
    def sample(self, feature, A):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]
        # at the very least watch at some gosh darn thing.
        # Yet why the model may refuse to focus is still a question to ask.
        normf = torch.clip(A.view(nB, nT, -1).sum(2).view(nB, nT, 1, 1), min=1)
        # Normalize
        A = A / normf
        # weighted sum
        C = self.getC(feature, A, nB, nC, nH, nW, nT)
        return A, C
        pass
