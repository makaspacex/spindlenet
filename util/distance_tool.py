import numpy as np
import matplotlib.pyplot as plt
import cv2

# import Levenshtein  as lvens
# print(lvens.distance("love","sffg"))
# print(lvens.distance("love","sffg", weights=(1,1,2)))
# print(lvens.distance("love","lovefghaa"))
# print(lvens.distance("love","lovefghaa", weights=(1,1,2)))

# from levenshtein_distance import Levenshtein
# lev_object = Levenshtein('test', 'text')
# distance = lev_object.distance()
# print(distance)
# print(lev_object.sequence_array())


class MakaLevenshtein:
    
    def __init__(self, str_a:str, str_b:str, weights:tuple =(1,1,1)) -> None:
        """
        Args:
            str_a (str): str_a
            str_b (str): _description_
            weights (tuple, optional): its order is  (insertion, deletion, substitution). Defaults to (1,1,1).
        """
        self.str_a = str_a
        self.str_b = str_b
        self.weights = weights
        
    
    def distance(self):
        
        str_a = self.str_a
        str_b = self.str_b
        
        ins_w, del_w, sub_w = self.weights
        # (insertion, deletion, substitution)
        
        str_a=str_a.lower()
        str_b=str_b.lower()
        
        matrix_ed=np.zeros((len(str_a)+1,len(str_b)+1), dtype=np.int8)
        matrix_ed[0]=np.arange(len(str_b)+1)
        matrix_ed[:,0] = np.arange(len(str_a) + 1)
        
        for i in range(1,len(str_a)+1):
            for j in range(1,len(str_b)+1):
                # 表示删除a_i
                dist_1 = matrix_ed[i - 1, j] + del_w
                # 表示插入b_i
                dist_2 = matrix_ed[i, j - 1] + ins_w
                # 表示替换b_i
                dist_3 = matrix_ed[i - 1, j - 1] + (sub_w if str_a[i - 1] != str_b[j - 1] else 0)
                
                #取最小距离
                matrix_ed[i,j]=np.min([dist_1, dist_2, dist_3])
        
        self. matrix_ed =  matrix_ed
        return matrix_ed[-1,-1]

    
    def numbers(self, use_weights=False):
        dp = self.matrix_ed
        i = len(dp) - 1
        j = len(dp[0]) - 1
        
        ins_w, del_w, sub_w = 1,1,1
        if use_weights:
            ins_w, del_w, sub_w = self.weights
        ins_e ,del_e,sub_e  = 0,0,0
        while i > 0 or j > 0:
            a = dp[i - 1][j - 1] if i > 0 and j > 0 else float("inf")
            b = dp[i - 1][j] if i > 0 else float("inf")
            c = dp[i][j - 1] if j > 0 else float("inf")
            min_val = min([a, b, c])

            if dp[i][j] == a and a == min_val:
                i -= 1
                j -= 1
                # 没有操作
            elif a == min([a, b, c]):
                #  通过替换来的
                i -= 1
                j -= 1
                sub_e += sub_w
            elif b == min([a, b, c]):
                i = i - 1
                del_e += del_w
            else:
                j = j - 1
                ins_e += ins_w
        
        return ins_e ,del_e,sub_e


if __name__ == "__main__":
    import Levenshtein  as lvens
    
    str1 = "wesds"
    str2 = "wesdddd"
    # (insertion, deletion, substitution)
    weights = (1,2,2)
    
    maka_leven = MakaLevenshtein(str1, str2, weights=weights )
    
    print(lvens.distance(str1, str2, weights=weights))

    print(maka_leven.distance())
    print(maka_leven.numbers())



