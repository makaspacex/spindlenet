from neko_sdk.AOF.neko_lens import neko_lens, vis_lenses
from neko_sdk.AOF.neko_reslayer_std import neko_reslayer_std, neko_reslayer_stdlr
from neko_sdk.encoders.ocr_networks.dan.dan_reslens_naive import DanResnet


class DanResnetStd(DanResnet):
    LAYER = neko_reslayer_std
    LENS = neko_lens


def res_naive_lens45_std(strides, compress_layer, hardness, inpch=1, oupch=512):
    model = DanResnetStd([3, 4, 6, 6, 3], strides, None, hardness, compress_layer, inpch=inpch, oupch=oupch)
    return model


def res_naive_lens45_std_thicc(strides, compress_layer, hardness, inpch=1, oupch=512):
    model = DanResnetStd([3, 4, 6, 6, 3], strides, None, hardness, compress_layer, inpch=inpch, oupch=oupch, expf=1.5)
    return model


class DanResnetStdlr(DanResnet):
    LAYER = neko_reslayer_stdlr
    LENS = neko_lens


def res_naive_lens45_stdlr(strides, compress_layer, hardness, inpch=1, oupch=512):
    model = DanResnetStdlr([3, 4, 6, 6, 3], strides, None, hardness, compress_layer, inpch=inpch, oupch=oupch)
    return model


def res_naive_lens45_stdlr_thicc(strides, compress_layer, hardness, inpch=1, oupch=512):
    model = DanResnetStdlr([3, 4, 6, 6, 3], strides, None, hardness, compress_layer, inpch=inpch, oupch=oupch,
                           expf=1.5)
    return model


if __name__ == '__main__':
    import torch
    import cv2
    import numpy as np

    im = cv2.resize(cv2.imread("/home/lasercat/Pictures/nos1080.png"), (256, 256))
    data = (torch.tensor(im).float() - 127) / 128  # C x H x W
    data = data.permute(2, 0, 1).unsqueeze(0)
    net = res_naive_lens45_std(**{
        'strides': [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
        'compress_layer': True,
        'inpch': 3,
        'hardness': 2
    })

    out, grid = net(data, True)
    oims = vis_lenses(data, grid)
    # out1, grid1 = net1(data, True)
    i = oims[0]
    img = (i.detach() * 127 + 127).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
    cv2.imwrite("/home/lasercat/tmp/lens_input" + str(0) + ".jpg", img)

    for iid in range(1, len(oims)):
        i = oims[iid]
        img = (i.detach() * 127 + 127).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
        cv2.imwrite("/home/lasercat/tmp/lens_after" + str(iid) + ".jpg", img)

    pass
