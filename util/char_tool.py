#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
desc: 初始化字典
author: 老马啸西风
date: 2021-11-24
'''
from pathlib import Path
class CharSimilarityHelper(object):
    
    def __init__(self) -> None:

        cur_dir = Path(__file__).resolve().parent
        
        # 字典初始化 
        self.bihuashuDict = self.initDict(cur_dir/ Path('db/bihuashu_2w.txt'));
        self.hanzijiegouDict = self.initDict(cur_dir/ Path('db/hanzijiegou_2w.txt'));
        self.pianpangbushouDict = self.initDict(cur_dir/ Path('db/pianpangbushou_2w.txt'));
        self.sijiaobianmaDict = self.initDict(cur_dir/ Path('db/sijiaobianma_2w.txt'));

        # 权重定义（可自行调整）
        self.hanzijiegouRate = 10;
        self.sijiaobianmaRate = 8;
        self.pianpangbushouRate = 6;
        self.bihuashuRate = 2;
        
    def initDict(self, path):
        dict = {}; 
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f.readlines():
                    # 移除换行符，并且根据空格拆分
                    splits = line.strip('\n').split(' ');
                    key = splits[0];
                    value = splits[1];
                    dict[key] = value; 
        return dict;
    
    # 计算核心方法
    '''
    desc: 笔画数相似度
    '''
    def bihuashuSimilar(self, charOne, charTwo): 
        valueOne = self.bihuashuDict[charOne];
        valueTwo = self.bihuashuDict[charTwo];
        
        numOne = int(valueOne);
        numTwo = int(valueTwo);
        
        diffVal = 1 - abs((numOne - numTwo) / max(numOne, numTwo));
        return self.bihuashuRate * diffVal * 1.0;

        
    '''
    desc: 汉字结构数相似度
    '''
    def hanzijiegouSimilar(self, charOne, charTwo): 
        valueOne = self.hanzijiegouDict[charOne];
        valueTwo = self.hanzijiegouDict[charTwo];
        
        if valueOne == valueTwo:
            # 后续可以优化为相近的结构
            return self.hanzijiegouRate * 1;
        return 0;
        
    '''
    desc: 四角编码相似度
    '''
    def sijiaobianmaSimilar(self, charOne, charTwo): 
        valueOne = self.sijiaobianmaDict[charOne];
        valueTwo = self.sijiaobianmaDict[charTwo];
        
        totalScore = 0.0;
        minLen = min(len(valueOne), len(valueTwo));
        
        for i in range(minLen):
            if valueOne[i] == valueTwo[i]:
                totalScore += 1.0;
        
        totalScore = totalScore / minLen * 1.0;
        return totalScore * self.sijiaobianmaRate;

    '''
    desc: 偏旁部首相似度
    '''
    def pianpangbushoutSimilar(self, charOne, charTwo): 
        valueOne = self.pianpangbushouDict[charOne];
        valueTwo = self.pianpangbushouDict[charTwo];
        
        if valueOne == valueTwo:
            # 后续可以优化为字的拆分
            return self.pianpangbushouRate * 1;
        return 0;  
        
    '''
    desc: 计算两个汉字的相似度
    '''
    def similar(self, charOne, charTwo):
        if charOne == charTwo:
            return 1.0;
        
        sijiaoScore = self.sijiaobianmaSimilar(charOne, charTwo);    
        jiegouScore = self.hanzijiegouSimilar(charOne, charTwo);
        bushouScore = self.pianpangbushoutSimilar(charOne, charTwo);
        bihuashuScore = self.bihuashuSimilar(charOne, charTwo);
        
        totalScore = sijiaoScore + jiegouScore + bushouScore + bihuashuScore;    
        totalRate = self.hanzijiegouRate + self.sijiaobianmaRate + self.pianpangbushouRate + self.bihuashuRate;
        
        
        result = totalScore*1.0 / totalRate * 1.0;
        
        # print('总分：' + str(totalScore) + ', 总权重: ' + str(totalRate) +', 结果:' + str(result));
        # print(f"权重 四角编码:{sijiaobianmaRate} 汉字结构:{hanzijiegouRate} 偏旁部首:{pianpangbushouRate} 笔画数:{bihuashuRate} ")
        # print(f"得分 四角编码:{sijiaoScore} 汉字结构:{jiegouScore} 偏旁部首:{bushouScore} 笔画数:{bihuashuScore} ")
        # 
        # print("")
        return result;

# 这里 末 未 相似度为1，因为没有拼音的差异。四角编码一致。
# 可以手动替换下面的字，或者读取文件，循环计算
'''
$ python main.py
总分：25.428571428571427, 总权重: 26, 结果:0.978021978021978
四角编码：8.0
汉字结构：10
偏旁部首：6
笔画数：1.4285714285714286
'''
def print_zh_fonts():
    from matplotlib.font_manager import FontManager
    import subprocess

    fm = FontManager()
    mat_fonts = set(f.name for f in fm.ttflist)

    output = subprocess.check_output('fc-list :lang=zh -f "%{family}\n"', shell=True)
    output = output.decode('utf-8')
    # print '*' * 10, '系统可用的中文字体', '*' * 10
    # print output
    zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n'))
    available = mat_fonts & zh_fonts

    print('*' * 10, '可用的字体', '*' * 10)
    for f in available:
        print(f)

if __name__ == "__main__":
    res = set()
    with open("chars/alphabet_TKHMTH2200_test_all.txt", 'r')  as f:
        content = f.read()
        chars = content.splitlines()
        res.update(chars)
    with open("chars/alphabet_TKHMTH2200_train.txt", 'r')  as f:
        content = f.read()
        chars = content.splitlines()
        res.update(chars)

    print(res)
    print(len(res))

    csh = CharSimilarityHelper()

    errs = set()
    for c in res:
        try:
            csh.similar(c, '来')
        except Exception:
            errs.update([c])
    print(errs)
    print(len(errs))
    # similar('末', '来')
    # similar('人', '入')
