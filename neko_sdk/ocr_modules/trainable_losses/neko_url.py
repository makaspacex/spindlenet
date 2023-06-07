import torch


# unknown ranking loss
class neko_unknown_ranking_loss:
    def onehot(self):
        pass

    def forward(self, pred, label, margin):
        with torch.no_grad():
            unk = pred.shape[1] - 1
            gtmask = torch.zeros_like(pred)
            gtmask = gtmask.scatter(1, label.unsqueeze(-1), 1)
            closest_err_idx = torch.argmax((pred - 9999 * gtmask)[:, :unk], dim=-1)
            valid = ((label != 0) * (label != unk)).float()
        corr = pred.gather(1, label.unsqueeze(-1)).NekoSqueeze(-1)
        err = pred.gather(1, closest_err_idx.unsqueeze(-1)).NekoSqueeze(-1)
        unks = pred[:, unk]
        bad_margins = torch.relu(unks - corr + margin) + torch.relu(err - unks + margin)
        return (bad_margins * valid).sum() / (valid.sum() + 1)
#
#
# if __name__ == '__main__':
#     l=neko_unknown_ranking_loss()
#     pred,label=torch.load("/home/lasercat/debug.pt")
#     loss=l.forward(pred,label,0.5)
#     print(loss)
