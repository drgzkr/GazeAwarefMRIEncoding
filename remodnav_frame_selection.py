import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import REMODNAV_DIR, FIXATIONS_DIR
import os

sub_list = ['sub-01','sub-02','sub-03','sub-04','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20']

for run in tqdm(['1','2','3','4','5','6','7','8']):
    
    for sub in sub_list:
        
        tsv_file = os.path.join(REMODNAV_DIR, sub+'-run'+run+'.tsv')
        
        #open tsv file in pandas dataframe
        with open(tsv_file, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            
        df = pd.read_csv(tsv_file, delimiter='\t')
        data = df.values
        
        #get coordinates at fixation and plot on frame
        event_type = 'FIXA'
        fixation_distilled = []
        for gaze_event in data:
            if gaze_event[2] == event_type:
                fix_x= np.mean([gaze_event[3],gaze_event[5]])
                fix_y=np.mean([gaze_event[4],gaze_event[6]])
                info = [gaze_event[0],fix_x,fix_y]
                
                fixation_distilled.append(info)
        fixation_distilled = np.array(fixation_distilled)
        fixation_frames = fixation_distilled[:,0]

        np.save(os.path.join(FIXATIONS_DIR, sub+'-run'+run+'.npy'), fixation_distilled)
        
        for fix in fixation_frames:
            frames_set.add(fix)
            
    frames_set = np.array(list(frames_set)).astype(int)
    np.save(os.path.join(FIXATIONS_DIR, 'Needed_frames_Run_'+run+'.npy'), frames_set)
