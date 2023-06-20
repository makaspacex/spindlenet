
import glob
from util.distance_tool import MakaLevenshtein

def main(eval_dir:str):
    res_path_list = glob.glob(f"{eval_dir}/*.txt")
    total_len = 0
    total_ins_e ,total_del_e, total_sub_e = 0,0,0
    total_dis =0
    
    for ii, p in enumerate(res_path_list):
        with open(p, "r") as f:
            lines = f.read().splitlines()
            
        gt, pred = lines[0],lines[1]
        total_len += len(gt)
        
        maka_leven = MakaLevenshtein(gt, pred)
        dis = maka_leven.distance()
        total_dis += dis
        
        ins_e ,del_e,sub_e = maka_leven.numbers()
        total_ins_e += ins_e
        total_del_e += del_e
        total_sub_e += sub_e
        
        cr = (total_len - total_del_e - total_sub_e) / total_len
        ar = (total_len - total_dis) / total_len
        
        print(f"{ii+1}/{len(res_path_list)} cr:{cr:.04f} ar:{ar:.04f}")
    

if __name__ == "__main__":
    # eval_dir = "runtime/OSTR_C2J_DTA_Only_MTH/mth_1200_new/logs_E1_1/closeset_benchmarks/MTH1200"
    eval_dir = "runtime/OSTR_C2J_DTA_Only_MTH/tkhmth2200_v1/logs_E1_tkhmth2200_test/closeset_benchmarks/TKHMTH2200"
    main(eval_dir=eval_dir)
    
    
    
    


