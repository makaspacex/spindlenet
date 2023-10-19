
import glob
from util.distance_tool import MakaLevenshtein

class MakaEval:
    def __init__(self) -> None:
        pass
        self.total_len = 0
        self.total_ins_e ,self.total_del_e, self.total_sub_e = 0,0,0
        self.total_ins_cs ,self.total_del_cs, self.total_sub_cs = [],[],[]
        self.total_dis =0

        self.total_samples=0
        self.acc_nums=0

        self.length_right_n,self.length_acc = 0,0
        
        self.ACC, self.AR, self.CR =0, 0,0
        
        self.all_preds = []
        self.all_gts = []

    def continue_eval(self, preds, gts, show_res=False):
        if not isinstance(preds,list):
            preds = [preds]
        if not isinstance(gts,list):
            gts = [gts]
        
        self.all_preds += preds
        self.all_gts += gts
        
        for pred, gt in zip(preds, gts):
            
            # 忽略#字符
            pred = pred.replace("#", "")
            gt = gt.replace("#", "")
            
            self.total_len += len(gt)
            self.total_samples += 1
            if len(pred) == len(gt):
                self.length_right_n += 1
            
            if pred == gt:
                self.acc_nums += 1
            
            maka_leven = MakaLevenshtein(pred, gt)
            dis = maka_leven.distance()
            self.total_dis += dis
            
            ins_e ,del_e,sub_e, ins_cs, del_cs, sub_cs = maka_leven.numbers()
            self.total_ins_e += ins_e
            self.total_del_e += del_e
            self.total_sub_e += sub_e
            
            self.total_ins_cs += ins_cs
            self.total_del_cs += del_cs
            self.total_sub_cs += sub_cs
        
        self.CR = (self.total_len - self.total_del_e - self.total_sub_e) / self.total_len
        self.AR = (self.total_len - self.total_dis) / self.total_len
        self.ACC = self.acc_nums/self.total_samples
        
        self.length_acc = self.length_right_n / self.total_samples

        self.res = {'CR':self.CR, "AR":self.AR, "ACC":self.ACC, 
                    "total_samples":self.total_samples,
                    "length_acc":self.length_acc,
                    "total_ins_cs": len(self.total_ins_cs),
                    "total_del_cs": len(self.total_del_cs),
                    "total_sub_cs": len(self.total_sub_cs)
                    }
        
        if show_res:
            self.show()
        
        return self.res
    
    def __str__(self) -> str:
        return f"TS:{self.total_samples} CR:{self.CR:.04f} AR:{self.AR:.04f} ACC:{self.ACC:.04f} L_ACC:{self.length_acc} L_ins_N:{len(self.total_ins_cs)} L_del_N:{len(self.total_del_cs)} L_sub_N:{len(self.total_sub_cs)}"
    
    def show(self):
        print(self)

def main(res_path_list:list):
    total_len = 0
    total_ins_e ,total_del_e, total_sub_e = 0,0,0
    total_dis =0
    maka_eval = MakaEval()
    
    for ii, p in enumerate(res_path_list):
        with open(p, "r") as f:
            lines = f.read().splitlines()
        
        gt, pred = lines[0],lines[1]
        maka_eval.continue_eval(pred,gt)
        
        total_len += len(gt)
        
        maka_leven = MakaLevenshtein(pred, gt)
        dis = maka_leven.distance()
        total_dis += dis
        
        ins_e ,del_e,sub_e = maka_leven.numbers()
        total_ins_e += ins_e
        total_del_e += del_e
        total_sub_e += sub_e
        
        cr = (total_len - total_del_e - total_sub_e) / total_len
        ar = (total_len - total_dis) / total_len
        
        # maka_eval.show()
        # print(f"{maka_eval.total_len} {maka_eval.total_del_e} {maka_eval.total_sub_e} {maka_eval.total_dis} ")
        print(f"{ii+1}/{len(res_path_list)} cr:{cr:.04f} ar:{ar:.04f} maka: {maka_eval}")

if __name__ == "__main__":
    # eval_dir = "runtime/OSTR_C2J_DTA_Only_MTH/mth_1200_new/logs_E1_1/closeset_benchmarks/MTH1200"
    eval_dir = "runtime/OSTR_C2J_DTA_Only_MTH/tkhmth2200/logs_E1_tkhmth2200_test/closeset_benchmarks/TKHMTH2200"
    res_path_list = glob.glob(f"{eval_dir}/*.txt")
    # res_path_list = res_path_list[:20]
    main(res_path_list)
    
    
    
    


