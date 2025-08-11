
import cv2
import re
import numpy as np
import os
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
root = os.path.dirname(os.path.realpath(__file__))
data_dir_ori = os.path.join(root, 'ModelTest_0330_ori')
data_dir_pro = os.path.join(root, 'ModelTest_0330_pro')


class features:
    ## 컬러값 
    @staticmethod
    def extract_rgb_from_filename(filename: str):

        match = re.search(r'_R(\d+)_G(\d+)_B(\d+)', filename)
        if match:
            return tuple(map(int, match.groups()))
        return None

    
    
    def RGB_Data(ProList):
        Allcount=len(ProList)
        Count=0
        rgb_list=[]

        for pro in ProList:
            rgb = features.extract_rgb_from_filename(pro)

            
            rgb_list.append(rgb)
            Count+=1

            print(f"{Count}/{Allcount}")


        return rgb_list


    
