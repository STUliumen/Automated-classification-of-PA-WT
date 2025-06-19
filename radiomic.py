#生成特征
import sys
import pandas as pd
import os
import random
import shutil
import numpy as np
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk

para_path = 'MR_1mm.yaml'
dir = 'features'

def radPA(name):
    extractor = featureextractor.RadiomicsFeatureExtractor(para_path)
    df = pd.DataFrame()
    features_dict = dict()
    path = os.path.join('nii', 'PA')
    path_seg = os.path.join('expand', 'PA')
    for folder in os.listdir(path):
        fname=os.path.splitext(folder)[0]
        fname = os.path.splitext(fname)[0]
        ori_path = os.path.join(path,folder)
        lab_path = os.path.join(path_seg,fname,fname+'_'+str(name)+'.nii.gz')
        try:
            features = extractor.execute(ori_path,lab_path)
        except:
            print(folder+" failed")
            print(ori_path+" "+lab_path)
            for key, value in features.items():
                features_dict[key] = -1
            features_dict['index'] = folder
            df = df._append(pd.DataFrame.from_dict(features_dict.values()).T, ignore_index=True)
        else:
            features_dict['index'] = folder
            for key, value in features.items():
                features_dict[key] = value
            df = df._append(pd.DataFrame.from_dict(features_dict.values()).T,ignore_index=True)
            print(folder+" success "+str(name))

    df.columns = features_dict.keys()
    df.to_csv(dir+'features_PA_'+str(name)+'.csv',index=0)


def radWT(name):
    extractor = featureextractor.RadiomicsFeatureExtractor(para_path)
    df = pd.DataFrame()
    features_dict = dict()
    path = os.path.join('nii', 'WT')
    path_seg = os.path.join('expand', 'WT')
    for folder in os.listdir(path):
        fname=os.path.splitext(folder)[0]
        fname = os.path.splitext(fname)[0]
        ori_path = os.path.join(path,folder)
        lab_path = os.path.join(path_seg,fname,fname+'_'+str(name)+'.nii.gz')
        try:
            features = extractor.execute(ori_path,lab_path)
        except:
            print(folder+" failed")
            print(ori_path+" "+lab_path)
            for key, value in features.items():
                features_dict[key] = -1
            features_dict['index'] = folder
            df = df._append(pd.DataFrame.from_dict(features_dict.values()).T, ignore_index=True)
        else:
            features_dict['index'] = folder
            for key, value in features.items():
                features_dict[key] = value
            df = df._append(pd.DataFrame.from_dict(features_dict.values()).T,ignore_index=True)
            print(folder+" success "+str(name))

    df.columns = features_dict.keys()
    df.to_csv(dir+'features_WT_'+str(name)+'.csv',index=0)



radPA('0')
radWT('0')