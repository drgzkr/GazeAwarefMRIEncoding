import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge # SAVE MODELS AND WEIGHTS
from scipy.stats import pearsonr
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

class FastRidge:
    def __init__(self, x_train, gamma=1.0):
        self._alpha   = None
        self._gamma   = gamma
        self._k_train = x_train @ x_train.T
        self._x_train = x_train

    def __call__(self, x_test):
        return x_test @ self._x_train.T @ self._alpha

    def estimate_alpha(self, y_train):
        self._alpha = np.linalg.pinv(self._k_train + self._gamma * np.eye(len(self._k_train), dtype=np.float32)) @ y_train

    def estimate_gamma(self, x_train, x_val, y_train, y_val, gammas=np.logspace(np.log10(0.1), np.log10(100_000_000), 10)):
        k_train = x_train @ x_train.T
        k_val   = x_val @ x_train.T
        r       = np.empty(len(gammas), np.float32)
        p       = np.empty(len(gammas), np.float32)

        for i in range(len(gammas)):
            alpha      = np.linalg.pinv(k_train + gammas[i] * np.eye(len(k_train), dtype=np.float32)) @ y_train
            y_val_test = k_val @ alpha
            r[i]       = ((y_val - y_val_test) ** 2).mean()
        self._gamma = gammas[np.argmin(r)]

    def get_weights(self):
        # Check if model has been trained
        if self._alpha is None:
            raise ValueError("Model hasn't been trained yet. Call estimate_alpha first.")

        # Transform dual coefficients (alpha) to primal weights
        weights = self._x_train.T @ self._alpha
    
        return weights

def test_train_splitter(run_lens,test_percent):
    # Given a list with number of timepoints each run and desired percentage of test data,
    # Returns indices for train-test split and train_train-train_test split for model validation
    # to be used on concaternated data

    half_test_percent = test_percent/2
    # create test and train indices by selecting mid %given of all runs as test
    run_lens_zero = np.insert(run_lens,0,0)
    indices = np.linspace(0,np.sum(run_lens)-1,np.sum(run_lens)).astype(int)
    #divide data into training and test set. Test is mid %given of each run
    test_indices = []
    run_train_lens = []
    for counter, run_len in enumerate(run_lens_zero):
        if counter >0:
            lower_value = int((run_len/2)-(run_len/half_test_percent))
            upper_value = int((run_len/2)+(run_len/half_test_percent))
            test_indices.append(indices[np.cumsum(run_lens_zero)[counter-1]+lower_value:np.cumsum(run_lens_zero)[counter-1]+upper_value])
            run_train_lens.append(run_len - indices[np.cumsum(run_lens_zero)[counter-1]+lower_value:np.cumsum(run_lens_zero)[counter-1]+upper_value].shape[0])
    test_indices = np.concatenate(test_indices).ravel()
    train_indices = np.delete(indices,test_indices)
    zero_run_train_lens = np.insert(run_train_lens,0,0)

    #split training data into test-train
    #take %given from the middle of each run
    train_indices_indices = np.linspace(0,train_indices.shape[0]-1,train_indices.shape[0]).astype(int)
    train_test_indices = []
    for counter, run_len in enumerate(zero_run_train_lens):
        if counter >0:
            lower_value = int((run_len/2)-(run_len/half_test_percent))
            upper_value = int((run_len/2)+(run_len/half_test_percent))
            train_test_indices.append(train_indices_indices[np.cumsum(zero_run_train_lens)[counter-1]+lower_value:np.cumsum(zero_run_train_lens)[counter-1]+upper_value])

    train_test_indices = np.concatenate(train_test_indices).ravel()
    train_train_indices = np.delete(train_indices_indices,train_test_indices)

    start_list = [0,86,174,261,358,451,539,648,715]
    stop_list =  [4,94,182,269,366,459,547,656,719]

    indices_to_remove = []
    for count, (start,stop) in enumerate(zip(start_list,stop_list)):
        indices_to_remove.append(test_indices[start:stop])
    indices_to_remove = np.concatenate(indices_to_remove).ravel()

    clipped_test_indices = test_indices[~np.isin(test_indices,indices_to_remove)]

    return train_indices, clipped_test_indices, train_train_indices, train_test_indices

run_lens = [451,441,438,488,462,439,542,338]


one_vis_degree_pixel = 1024/20

train_indices, test_indices, train_train_indices, train_test_indices = test_train_splitter(run_lens,20)


sub_list = ['sub-01','sub-02','sub-03','sub-04','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20']
all_subs_results = []
all_sub_PCA_exp_var = []
gammas = []
for sub in tqdm(sub_list):
    # disable warning logs
    print(sub)
    warnings.filterwarnings('ignore')

    # x and y sizes of feature maps to be used. Can change each flexibly
    feat_map_size = np.array([7,16])
    #sub = sub_list[sub_count]

    # load features per frames and prf estimates per voxel
    sub_features = np.load('/home/djaoet/wrkgrp/Dora/Study_Forrest/vgg19/HyperlayersBySubsRemodnav/Model_features_'+sub+'_hyperlayer_7_16.npy',allow_pickle=True)
    print('features loaded')
    # load brain data
    sub_movie_data = np.load('/home/djaoet/wrkgrp/Dora/Study_Forrest/Movie_Data/normalized_masked_concat_movie_'+sub+'.npy',allow_pickle=True)
    print('fMRI data loaded')
    # load concatenated fixation frames and coordinates, and collect some info about runs
    concat_fixations = np.zeros((sub_features.shape[0],3))
    run_fix_lens = []
    run_sub_frames = []
    counter = 0

    for run in ['1','2','3','4','5','6','7','8']:
        run_fixations = np.load('/home/djaoet/wrkgrp/Dora/Study_Forrest/Rect Alexnet/Frames_Per_Sub_Remodnav/Needed_frames_'+sub+'-run'+run+'.npy',allow_pickle=True)
        concat_fixations[counter:counter+run_fixations.shape[0]] = run_fixations
        run_fix_lens.append(run_fixations.shape[0])
        run_sub_frames.append(run_fixations[:,0].astype(int))
        counter += run_fixations.shape[0]


    # create empty average feature model brain
    average_feat_model_brain = np.zeros((sub_movie_data.shape[0],3599))
    # create empty result log array
    output_summary = np.zeros((sub_movie_data.shape[0],14,3))
    # create NAN FIXATION logs and EMPTY TRS logs
    all_voxels_all_nan_fixations = []
    all_voxels_empty_trs = []


    # create model weight logs
    encoder_model_brain_weights = np.zeros((sub_movie_data.shape[0],164864)) #FIND OUT was sub_features.shape[1]

    # grab the necessary features from necessary frames all at once
    ##### CENTER FIXATION 
    # get fixations at all frames
    concat_fixations_x = np.zeros_like(concat_fixations[:,1])
    concat_fixations_y = np.zeros_like(concat_fixations[:,2])
    concat_fixations_x[:] = (1280/2)
    concat_fixations_y[:] = (544/2)
    
    ###################### X coordinate ####################
    # give 50 pixels allowance for the final result x
    concat_fixations_x[np.where((concat_fixations_x>= 1280) & (concat_fixations_x <=1330))[0]] = 1279
    concat_fixations_x[np.where((concat_fixations_x >= -50) & (concat_fixations_x <= 0))] = 0
    # convert to layer map coordinates x
    concat_fixations_x = concat_fixations_x/(1280/feat_map_size[1])
    concat_fixations_x = concat_fixations_x.astype(int)
    # check nans x
    x_fixation_nan_indices = np.where((concat_fixations_x>= feat_map_size[1]) | (concat_fixations_x <0))[0]
    concat_fixations_x[x_fixation_nan_indices] = 0

    ###################### Y coordinate ####################
    # give 50 pixels allowance for the final result y 
    concat_fixations_y[np.where((concat_fixations_y >= 544) & (concat_fixations_y <= 594))[0]] = 543
    concat_fixations_y[np.where((concat_fixations_y >= -50) & (concat_fixations_y <= 0))] = 0
    # convert to layer map coordinates y
    concat_fixations_y = concat_fixations_y/(544/feat_map_size[0])
    concat_fixations_y = concat_fixations_y.astype(int)
    # check nans y
    y_fixation_nan_indices = np.where((concat_fixations_y>= feat_map_size[0]) | (concat_fixations_y <0))[0]
    concat_fixations_y[y_fixation_nan_indices] = 0

    # determine all timepoints where sample falls outside of frame
    all_nan_indices = np.unique(np.concatenate((x_fixation_nan_indices,y_fixation_nan_indices)))
    all_voxels_all_nan_fixations.append(all_nan_indices)
    # set the x-y coordinates to sample to fill in the blanks
    concat_fixations_x[x_fixation_nan_indices] = 0
    concat_fixations_y[y_fixation_nan_indices] = 0
    
    # grab the necessary features from necessary frames all at once
    voxel_feature_timeseries = sub_features[np.linspace(0,concat_fixations_x.shape[0]-1,concat_fixations_x.shape[0]).astype(int),:,concat_fixations_y,concat_fixations_x]

    #set the nan indices to 0
    voxel_feature_timeseries[all_nan_indices] = 0
    #set the nan indices to 0
    # select an hrf delay in frames (25frames/1sec)
    hrf_frame_offset = 112

    #downsample voxel feature timeseries to TR's
    voxel_feature_timeseries_tr = np.zeros((3599,voxel_feature_timeseries.shape[1]))
    empty_trs = []
    fix_count = 0
    run_tr_count = 0
    for run,run_len,run_fix_len,sub_needed_frames in zip(['1','2','3','4','5','6','7','8'],[451,441,438,488,462,439,542,338],run_fix_lens,run_sub_frames):

        sub_needed_frames_hrf = sub_needed_frames + hrf_frame_offset
        run_voxel_feature_timeseries = voxel_feature_timeseries[fix_count:fix_count+run_fix_len]

        #for each tr in run
        for num_tr in range(run_len):
            # get the frame numbers in tr
            tr_frames = np.where(np.floor(sub_needed_frames_hrf/50).astype(int)==num_tr)[0]

            # check if there are tr's with no frame
            if tr_frames.shape[0]>0: # if tr has frames, average them to get 1 tr

                tr_average = np.nanmean(run_voxel_feature_timeseries[tr_frames],axis=0)

            else: # if not, set them to 0 or mean, decided on mean, could change later

                # identify and save empty tr's for future check, requires manual separation of runs!
                empty_trs.append(num_tr)

                filler = np.empty((run_voxel_feature_timeseries.shape[1]))
                filler[:] = np.mean(run_voxel_feature_timeseries,axis=0)
                tr_average = filler
            # place the tr in big array
            voxel_feature_timeseries_tr[run_tr_count+num_tr]= tr_average

        fix_count += run_fix_len
        run_tr_count += run_len


    # add list of empty trs to bigger log
    all_voxels_empty_trs.append(empty_trs)

    # lengths of runs to normalize each run separately
    zero_run_lens = [0,451,441,438,488,462,439,542,338]

    # normalize feature data
    # create empty array to for normalized timeseries
    normalized_voxel_feature_timeseries_tr = np.zeros_like(voxel_feature_timeseries_tr.T)

    #for each feature in full timeseries
    for feature_counter, feature in enumerate(voxel_feature_timeseries_tr.T):
        #for each run in feature
        for run_count, run in enumerate(range(8)):
            #get indice where run starts and ends
            run_start = np.cumsum(zero_run_lens)[run_count]
            run_end = np.cumsum(zero_run_lens)[run_count+1]
            #grab run feature timeseries
            run_feature = feature[run_start:run_end]

            # if timeseries is not filled with zeros, and if there are more than 1 non-nan value,
            if (np.nansum(run_feature) != 0) & ((run_feature.shape[0] - np.sum(np.isnan(run_feature)))>1):
                #get log +1 of run timeseries
                norm_feature = np.log1p(run_feature)
                #get mean of run timeseries
                feature_timecourse_mean_1 = np.nanmean(norm_feature)
                #substract mean from each timepoint
                norm_feature = norm_feature - feature_timecourse_mean_1
                #get standart deviation of run timeseries
                feature_std = np.nanstd(norm_feature)
                #if there is no variation,
                if feature_std == 0:
                    #set normalized run timeseries to mean (doesnt matter as they are all same value)
                    norm_feature[:] = feature_timecourse_mean_1
                #if there is variation,
                else:
                    # divide mean substracted timeseries by std
                    norm_feature = np.divide(norm_feature,feature_std)
                # put normalized run timeseries into big array
                normalized_voxel_feature_timeseries_tr[feature_counter,run_start:run_end] = norm_feature
            else:
                # its already filled with 0s, no normalization needed
                normalized_voxel_feature_timeseries_tr[feature_counter,run_start:run_end] = run_feature
    # transpose normalized feature timeseries
    normalized_voxel_feature_timeseries_tr = normalized_voxel_feature_timeseries_tr.T
    print('fMRI data normalized')

    #test train features
    f_te = normalized_voxel_feature_timeseries_tr[test_indices] # features.reshape(n_te, -1)
    f_tr = normalized_voxel_feature_timeseries_tr[train_indices]# features.reshape(n_tr, -1)
    f_tr_tr = f_tr[train_train_indices]# features.reshape(n_tr, -1)
    f_tr_te = f_tr[train_test_indices]# features.reshape(n_tr, -1)

    #test train brain data
    x_te = sub_movie_data[:,test_indices].T # brain responses, test set
    x_tr = sub_movie_data[:,train_indices].T# brain responses, training set
    x_tr_tr = x_tr[train_train_indices]# brain responses, training set
    x_tr_te = x_tr[train_test_indices]# brain responses, training set

    print('training baseline encoder')
    # Fast Ridge
    results = np.zeros((19629,2))

    fastRidge = FastRidge(f_tr_tr)
    try:
        fastRidge.estimate_gamma(f_tr_tr, f_tr_te, x_tr_tr, x_tr_te)
        fastRidge.estimate_alpha(x_tr_tr)

        print('encoder trained')
        y_test_hat = fastRidge(f_te)

        for voxel_count, (predicted, real) in enumerate(zip(y_test_hat.T,x_te.T)):
            if ~np.isnan(predicted).any():
                results[voxel_count,0], results[voxel_count,1] = pearsonr(predicted,real)
        print(np.mean(results[:,0]))
        all_subs_results.append(results)

        best_gamma = fastRidge._gamma
        print(best_gamma)
        gammas.append(best_gamma)

        np.save('/home/djaoet/wrkgrp/Dora/Study_Forrest/FastBaselineResults/'+str(sub)+'_centergaze_hyperlayer_fast_baseline_results',results,allow_pickle=True)
        print('results saved')
    except:
        print('exception occured 🤷‍')
np.save('/home/djaoet/wrkgrp/Dora/Study_Forrest/FastBaselineResults/'+str(sub)+'_centergaze_hyperlayer_fast_baseline_gammas',gammas,allow_pickle=True)
