#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:17:45 2022

@author: dorgoz
"""
import numpy as np
from tqdm import tqdm

sub_list = ['sub-01','sub-02','sub-03','sub-04','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20']

for sub in tqdm(sub_list):
    
    sub_features_shape = np.load('/home/djaoet/wrkgrp/Dora/Study_Forrest/vgg19/FeaturesBySubsRemodnav/Model_Features_'+sub+'_concat_run_layer0_7_16.npy').shape
    sub_feature_sizes = np.array([64,128,256,512,512])
    zero_sub_feature_sizes = np.insert(np.cumsum(sub_feature_sizes),0,0)
    
    sub_hyperlayer_features = np.zeros((sub_features_shape[0],np.sum(sub_feature_sizes),7,16))
    
    for layer in ['0','1','2','3','4']:
        sub_features = np.load('/home/djaoet/wrkgrp/Dora/Study_Forrest/vgg19/FeaturesBySubsRemodnav/Model_Features_'+sub+'_concat_run_layer'+layer+'_7_16.npy')
        sub_hyperlayer_features[:,zero_sub_feature_sizes[int(layer)]:zero_sub_feature_sizes[int(layer)+1],:,:] = sub_features
    
    np.save('/home/djaoet/wrkgrp/Dora/Study_Forrest/vgg19/HyperlayersBySubsRemodnav/Model_features_'+sub+'_hyperlayer_7_16',sub_hyperlayer_features,allow_pickle=True)
