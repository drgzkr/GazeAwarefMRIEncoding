#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 08:45:36 2022

@author: dorgoz
"""

import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from config import FIXATIONS_DIR, VGG_BY_RUN_DIR, VGG_BY_SUB_DIR


def model_features_sub(sub_id):
    sub_list = ['sub-01','sub-02','sub-03','sub-04','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20']

    sub = sub_list[sub_id]
    for layer in tqdm(['0','1','2','3','4']):
    #for sub in tqdm(sub_list):
        # get number of frames of sub for empty array
        features_shape = np.load(os.path.join(VGG_BY_RUN_DIR, 'Run8_Layer'+layer+'_vgg19_features_resized_to_7_16.npy'), allow_pickle=True).shape

        concat_frames_num = 0
        for run in ['1','2','3','4','5','6','7','8']:
           sub_needed_frames_size = np.load(os.path.join(FIXATIONS_DIR, sub+'-run'+run+'.npy'))[:,0].astype(int).shape[0]
           print(sub_needed_frames_size)
           concat_frames_num += sub_needed_frames_size
           
        all_runs_concat_features = np.empty((concat_frames_num,features_shape[1],features_shape[2],features_shape[3]))
        
        counter = 0
        for run,run_len in zip(['1','2','3','4','5','6','7','8'],[451,441,438,488,462,439,542,338]):
            #load the features of the entire run for all subs
            features = np.load(os.path.join(VGG_BY_RUN_DIR, 'Run'+run+'_Layer'+layer+'_vgg19_features_resized_to_7_16.npy'), allow_pickle=True)
            
            #load frame ids for the whole run and specific sub
            all_needed_frames = np.load(os.path.join(FIXATIONS_DIR, 'Needed_frames_Run_'+run+'.npy'))    
            sub_needed_frames = np.load(os.path.join(FIXATIONS_DIR, sub+'-run'+run+'.npy'))[:,0].astype(int)
            
            # for subject, get the indices of the needed features from list feature index by frame
            sub_frames_mask=np.zeros_like(sub_needed_frames)
            for frame_count, frame in enumerate(sub_needed_frames):
                sub_frames_mask[frame_count] = np.where(all_needed_frames==frame)[0][0]
            
            #get needed frames from whole array and put it in concat array for subject
            sub_features = features[sub_frames_mask]
            
            all_runs_concat_features[counter:counter+sub_features.shape[0]] = sub_features
            counter += sub_features.shape[0]
            
        np.save(os.path.join(VGG_BY_SUB_DIR, 'Model_Features_'+sub+'_concat_run_layer'+layer+'_7_16.npy'), all_runs_concat_features, allow_pickle=True)

Parallel(n_jobs=5)(delayed(model_features_sub)(sub_id)for sub_id in [0,1,2,3,4,5,6,7,8,9,10,11,12])  

