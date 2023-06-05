from neko_sdk.ocr_modules.renderlite.lib_render import RenderLite
from neko_sdk.ocr_modules.renderlite.addfffh import refactor_meta,add_masters,finalize
import torch;
#
# servants="QWERTYUIOPASDFGHJKLZXCVBNM";
# masters="qwertyuiopasdfghjklzxcvbnm";
# another example to make it fancier(multiple centers)
# servants=["qf1","wf2",'qf2','wf1'];
# masters="qwqw";


from neko_sdk.ocr_modules.charset.chs_cset import t1_3755;


def make_chlat_mc_dict(font,protodst):
    # cpx=[];
    # for i in chs37552cpx.values():
    #     cpx+=i;
    chrset=list(t1_3755)+list(set("QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm1234567890"));
    chrset=list(set(chrset));
    # masterschs=[];
    # servants_cpx=[];
    # for i in chs37552cpx:
    #     for j in chs37552cpx[i]:
    #         masterschs.append(i)
    #         servants_cpx.append(j);

    engine = RenderLite(os=84,fos=32);
    font_ids=[0 for c in chrset];
    meta=engine.render_core(chrset,['[s]'],font,font_ids,False);
    meta=refactor_meta(meta,unk=len(chrset)+len(['[s]']));
    # inject a shapeless UNK.
    servants="QWERTYUIOPASDFGHJKLZXCVBNM";
    masters="qwertyuiopasdfghjklzxcvbnm";
    meta["protos"].append(None)
    meta["achars"].append("[UNK]")

    # add_masters(meta,servants,masters);
    meta=add_masters(meta,servants,masters);
    # meta=add_masters(meta,servants_cpx,masterschs);
    meta=finalize(meta);
    print(protodst)
    torch.save(meta,protodst);
