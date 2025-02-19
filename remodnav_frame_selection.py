import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

save_path = '/path/Frames_Per_Sub_Remodnav/'

sub_list = []


for run in tqdm(['1','2','3','4','5','6','7','8']):
    
    for sub in sub_list:
        
        tsv_file = '/path/Remodnav/'+sub+'-run'+run+'.tsv'
        
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

        np.save(save_path+sub+'_frames_used_in_sec_run_'+run,fixation_distilled)
        
        for fix in fixation_frames:
            frames_set.add(fix)
            
    frames_set = np.array(list(frames_set)).astype(int)
    np.save(save_path+'Needed_frames_Run_'+run,frames_set)
