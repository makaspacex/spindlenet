from maka_ocr.data.processes.data_process import DataProcess
import numpy as np
import cv2


class MakeImgCenterAndPad(DataProcess):
    """_summary_
        竖条状的图像，居中贴图，并且resieze到目标大小
    Args:
        DataProcess (_type_): _description_
    """
    
    def __init__(self, img_hw: list) -> None:
        super().__init__()
        self.img_hw = img_hw
        
    def resize_v_align(self, cur_ratio,target_ratio,img_height,img_width):
        if cur_ratio < target_ratio:
            cur_target_height=img_height;
            # print("if", cur_ratio, self.target_ratio)
            cur_target_width = int(img_height * cur_ratio);
        else:
            cur_target_width = img_width
            cur_target_height = int(img_width/cur_ratio);
        return cur_target_height,cur_target_width;
 
    def __call__(self, data):
        img = data['image']
        c_img_h, c_img_w = img.shape[:2]
        img_h, img_w = self.img_hw
        
        cur_ratio = c_img_w / c_img_h
        target_ratio = img_w / img_h
        
        mask_height = img_h
        mask_width = img_w
        img = np.array(img)

        if (len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        cur_target_height,cur_target_width=self.resize_v_align(cur_ratio,target_ratio,img_h,img_w)
        
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        start_x = int((mask_height - img.shape[0]) / 2)
        start_y = int((mask_width - img.shape[1]) / 2)
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
        bmask = np.zeros([mask_height, mask_width]).astype(np.float32)
        bmask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = 1
        img = mask
        
        data['image'] = img
        data['bmask'] = bmask
        
        return data
    

