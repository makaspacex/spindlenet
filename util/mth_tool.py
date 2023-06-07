import glob
import opencc
import copy
from typing import List, Dict, Tuple, Optional,Union
from fontTools.ttLib import TTFont
from pathlib import Path
import cv2
from PIL import ImageFont, Image, ImageDraw
import textwrap
import glob
from tqdm.notebook import tqdm
import os
from neko_sdk.ocr_modules.charset.etc_cset import latin62 

def make_alphabet(text_file_list,dest_file_path):
    res_dict = {}
    res_set = set()
    converter = opencc.OpenCC('s2t.json');
    conv_t2s = opencc.OpenCC('t2s.json');
    upr="QWERTYUIOPASDFGHJKLZXCVBNM";
    lwr="qwertyuiopasdfghjklzxcvbnm";
    dig="1234567890";
    for t_path  in text_file_list:
        with open(t_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                a = line.split(" ")[0]
                b = conv_t2s.convert(a);
                for cid in range(len(b)):
                    k=b[cid];
                    v=a[cid];
                    if(k in latin62 or k in dig):
                        continue;
                    if(k not in res_dict):
                        res_dict[k]=set([k]);
                    res_dict[k].add(v);
    for i in range(26):
        res_dict[upr[i]]=set([upr[i],lwr[i]]);
    for i in range(10):
        res_dict[dig[i]]=set(dig[i]);
    
    with open(dest_file_path, 'w+') as f:
        for key, ele in res_dict.items():
            ele=ele.difference(set([key]));
            _x = key+" "+" ".join(list(ele))
            f.write(f"{_x}\n")

def make_alphabet_v2(text_file_list,dest_file_path):
    # 所有字符编程单列
    res_dict = {}
    ddirs = os.path.dirname(dest_file_path)
    os.makedirs(ddirs, exist_ok=True)
    
    # upr="QWERTYUIOPASDFGHJKLZXCVBNM"
    # all_str = upr + upr.lower()
    # for cc in all_str:
    #     if cc not in res_dict:
    #         res_dict[cc] = 0
    #     res_dict[cc] += 1
    
    # dig="1234567890"
    # for cc in dig:
    #     if cc not in res_dict:
    #         res_dict[cc] = 0
    #     res_dict[cc] += 1
    
    for t_path  in text_file_list:
        with open(t_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                a = line.split(",")[0]
                for cc in a:
                    if cc not in res_dict:
                        res_dict[cc] = 0
                    res_dict[cc] += 1
    
    with open(dest_file_path, 'w+') as f:
        for c, _ in res_dict.items():
            f.write(f"{c}\n")
        
    return res_dict

def get_char_set(txt_f_paths):
    if not isinstance(txt_f_paths, list):
        txt_f_paths = [txt_f_paths]
    
    alphabets = set()
    for ffp in txt_f_paths:
        with open(ffp) as f:
            content = f.read()
        content = content.replace("\n","").replace(" ","")
        alphabets.update(list(content))
    return alphabets


def find_char_in_fonts(char, ttfonts: Union[List[TTFont], TTFont]):
    if not isinstance(ttfonts, list):
        ttfonts = [ttfonts]
    
    for ii, font in enumerate(ttfonts):
        ttFont = font['cmap'].tables[0].ttFont
        uniMap = ttFont.getBestCmap()
        ord_c = ord(char)
        hex_c = hex(ord_c)[2:]
        hex_c = str(hex_c).upper()
        if ord_c in uniMap.keys():
            return ii, font
    return None, None


def get_char_img_from_label(res_dict = None):
    # img_paths = glob.glob("./ddata/TKHMTH2200/*/img/*")
    if res_dict is None:
        res_dict = {}
    
    char_gt_paths = glob.glob("/home/izhangxm/Desktop/work/main/VSDF/ddata/TKHMTH2200/*/label_char/*.txt")
    char_gt_paths = char_gt_paths
    
    pbar = tqdm(total=len(char_gt_paths))
    
    for ii, char_gt_path in enumerate(char_gt_paths):
        char_gt_path = Path(char_gt_path)
        pbar.update(1)
        pbar.set_description(f'Processing:{ii+1}/{len(char_gt_paths)}')
        with open(char_gt_path, "r") as f:
            lines = f.readlines()
        
        read_img = False
        for line in lines:
            eles = line.replace("\n","").split(" ")
            if len(eles) != 5:
                continue
            char = eles[0]
            if char in res_dict:
                continue
            read_img = True
        
        if not read_img:
            continue
        
        try:
            img_path = char_gt_path.parent.parent/"img"/f"{char_gt_path.stem}.jpg"
            if not img_path.exists():
                img_path = char_gt_path.parent.parent/"img"/f"{char_gt_path.stem}.png"
            if not img_path.exists():
                raise Exception(f"not exist: {img_path}")
            img = cv2.imread(str(img_path))
        except Exception as e:
            print(e)
            continue
        
        try:
            for line in lines:
                eles = line.replace("\n","").split(" ")
                if len(eles) != 5:
                    continue
                char = eles[0]
                x1,y1,x2,y2 = [int(float(_)) for _ in eles[1:]]
                if char in res_dict:
                    continue
                char_img = img[y1:y2, x1:x2,:]
                char_img = cv2.resize(char_img, (64, 64), interpolation=cv2.INTER_LINEAR)
                
                res_dict[char] = char_img
        
        except Exception as e:
            print(char_gt_path)
            raise e
    
    return res_dict

import os
import regex
from osocr_tasks.tasksg1.dscs import makept_for_MTH

def make_dict(aplphabets_file_path, font_paths, dst_path, force_rebuild=False):
    if(os.path.exists(dst_path)) and (not force_rebuild):
        print("skipping pt build, to force rebuilding, remove", dst_path);
        return dst_path;
    with open(aplphabets_file_path, 'r') as fp:
        chars=[l.strip() for l in fp];
        allch=[];
        masters=[];
        servants=[];
        for ch in chars:
            if(len(ch)):
                l= regex.findall(r'\X', ch, regex.U);
                allch+=l;
                for i in range(1,len(l)):
                    masters.append(l[0]);
                    servants.append(l[i]);
    allch=set(allch)
    makept_for_MTH(None,font_paths,dst_path,allch, {}, masters=masters, servants=servants);
    return dst_path

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import ImLmdbWrapper
import glob
import os.path
import cv2
from pathlib import Path

def cal_bounding_box(ps):
    # 左下角，顺时针转圈, 返回左上角+右下角的信息
    x1,y1,x2,y2,x3,y3,x4,y4 =  ps
    _x1 = min(x1,x2,x3,x4)
    _y1 = min(y1,y2,y3,y4)
    _x2 = max(x1,x2,x3,x4)
    _y2 = max(y1,y2,y3,y4)
    return _x1,_y1,_x2,_y2

def quick_lmdb_for_mth(gt_file_path_list, dst, lang="None"):
    if os.path.exists(f"{dst}/data.mdb"):
        os.remove(f"{dst}/data.mdb")
    if os.path.exists(f"{dst}/lock.mdb"):
        os.remove(f"{dst}/lock.mdb")
    db=ImLmdbWrapper(dst)
    for ii, char_file in enumerate(gt_file_path_list):
        char_file = Path(char_file)
        if ii % 500 == 0:
            print(f"{ii}/{len(gt_file_path_list)} {char_file}")
        mth_dataset_dir = str(Path(char_file).parent.parent)
        img_dir = Path(os.path.join(mth_dataset_dir,"img"))
        img_file_path = img_dir/ f"{char_file.stem}.png"
        
        if not img_file_path.exists():
            img_file_path = img_dir/ f"{char_file.stem}.jpg"
        
        if not img_file_path.exists():
            print(img_file_path)
            continue
        
        img_full = cv2.imread(str(img_file_path))
        try:
            f = open(char_file, 'r')
            lines = f.readlines()
            for line in lines:
                eles = line.replace("\n", "").split(',')
                anno = eles[0]
                ps = [int(_) for _ in eles[1:]]
                x1,y1,x2,y2 = cal_bounding_box(ps)
                img = img_full[y1:y2, x1:x2,:]
                db.add_data_utf(img,anno,lang)
        except Exception as e:
            print(e, line)
            print(img_file_path)
        finally:
            f.close()
    db.end_this()
def split_res_dict_for_matplot(db_alphabet_res_dict: dict):
    X = []
    Y = []
    for c, n  in db_alphabet_res_dict.items():
        X.append(c)
        Y.append(n)
    return X, Y

def process_mth_path(img_path_list, mth_dir):
    new_list = []
    for p in img_path_list:
        p = Path(p.replace('\n',''))
        ele = p.parts
        r = f"{mth_dir}/{ele[-3]}/label_textline/{p.stem}.txt"
        new_list.append(r)
        if not os.path.exists(r):
            print("+++++++", r)
    return new_list


def render_char(alphbats_file_list, img_save_path, font_path_list):
    # txt
    alphabets = get_char_set(alphbats_file_list)
    alphabets = list(alphabets)
    text = "".join(alphabets)

    # fonts

    fontSize = 64
    ttfonts,imFonts = [],[]
    for f_p in font_path_list:
        ttfonts.append( TTFont(f_p))
        imFonts.append(ImageFont.truetype(f_p, size=fontSize))

    font_family = ImageFont.truetype_family(*imFonts)

    char_font_ids = []
    for char in alphabets:
        ii, _ = find_char_in_fonts(char, ttfonts=ttfonts)
        char_font_ids.append(ii)

    fontSize = 64
    max_len = 32
    lines = textwrap.wrap(text, width=max_len)

    FOREGROUND = "#000000"
    bground = (255,255,255)

    im_font = imFonts[0]
    tw, th = im_font.getsize(char)
    im = Image.new("RGB", ((fontSize * max_len), len(lines) * (fontSize + 8)), (255, 255, 255))
    dr = ImageDraw.Draw(im)

    miss_char = []
    x,y=0,0
    for i,char,f_id in zip(range(len(alphabets)), alphabets, char_font_ids):
        if f_id is None:
            col = i % max_len
            row = int(i/max_len)
            miss_char.append(f"{row+1},{col}, {char}")
            f_id = 0
        im_font = imFonts[f_id]

        x = tw * (i % max_len)
        y = th * int(i/max_len)
        
        dr.text((x, y), char, font=im_font, fill=FOREGROUND, spacing=0, align="left")
        
    print(miss_char)
    im.save(img_save_path)


def eval_fonts(char_set, font_path_list):
    char_in_unimap = set()
    miss_char = copy.deepcopy(char_set)

    for f_p in font_path_list:
        font = TTFont(f_p)
        ttFont = font['cmap'].tables[0].ttFont
        uniMap = ttFont.getBestCmap()
        glyfMapDict = font['glyf']
        
        _new_miss = []
        for c in miss_char:
            ord_c = ord(c)
            hex_c = hex(ord_c)[2:]
            hex_c = str(hex_c).upper()
            if (f"u{hex_c}" in glyfMapDict) or (f"uni{hex_c}" in glyfMapDict):
                if c in char_in_unimap:
                    char_in_unimap.remove(c)
                continue
            
            if ord_c in uniMap.keys():
                char_in_unimap.add(c)
            
            _new_miss.append(c)
            
            # 方法1 总计6631 检测出331个缺失
            # try:
            #     ttFont.getGlyphID(f"uni{str(hex_c).upper()}")
            # except Exception:
            #     _new_miss.append(c)
            #     continue
            # 方法2 总计6631 检测出3个缺失
            # if ord_c not in uniMap.keys():
            #     _new_miss.append(c)
            #     continue
        miss_char = _new_miss

    print(f"miss: {len(miss_char)} total:{len(char_set)}")
    # print([f"char:{_}, ucode:{hex(ord(_))}" for _ in miss_char])
    print(f"cha not in glyph({len(miss_char)}): {miss_char}")
    print(f"cha not in glyph but in uniMap({len(char_in_unimap)}): {char_in_unimap}")
    print(f"miss: {set(miss_char) - char_in_unimap}")
    # print(hex(ord("嗢")))


DB_NAMES = ["MTH1000", "MTH1200", "TKH"]
