# python_argument.py

# ===== Global imports =====
import argparse
import time
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge # SAVE MODELS AND WEIGHTS
from sklearn.linear_model import RidgeCV
import os
from config import FIXATIONS_DIR, HYPERLAYERS_DIR, FMRI_DIR, PRF_DIR, RESULTS_DIR
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import warnings
import pickle


# ===== Global variables =====
sub_list = ['sub-01','sub-02','sub-03','sub-04','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20']
run_lens = [451,441,438,488,462,439,542,338]

one_vis_degree_pixel = 1024/20

test_percent = 20

# ===== Functions =====
class ModifiedKernelRidgeCV:
    def __init__(self):
        self.target = None
        self.val_size = None

    def train(self, f_train, f_val, f_full, x_train, x_val, x_full, alphas=None):
        """
        f_*: feature matrices (kernels or features)
        x_*: target vectors
        alphas: list of coarse search ridge parameters (if None, default used)
        """
        if alphas is None:
            coarse_exponents = np.linspace(-1, 8, 10)
            alphas = 10.0 ** coarse_exponents
        else:
            alphas = np.array(alphas)

        ridge_cv = RidgeCV(alphas=alphas, cv=None)
        ridge_cv.fit(f_full, x_full)
        y = ridge_cv.predict(f_full)

        full_training_summary = {'model':ridge_cv,
                                'pearsonr':pearsonr(x_full, y)[0],
                                'pearsonrp':pearsonr(x_full, y)[1]
                                }

        return full_training_summary



def test_train_splitter(run_lens,test_percent):
    # Given a list with number of timepoints each run and desired percentage of test data,
    # Returns indices for train-test split and train_train-train_test split for model validation
    # to be used on concatenated data
    
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

    #make properly next time!
    start_list = [0,86,174,261,358,451,539,648,715]    
    stop_list =  [4,94,182,269,366,459,547,656,719]
    
    indices_to_remove = []
    for count, (start,stop) in enumerate(zip(start_list,stop_list)):
        indices_to_remove.append(test_indices[start:stop])
    indices_to_remove = np.concatenate(indices_to_remove).ravel()
    
    clipped_test_indices = test_indices[~np.in1d(test_indices,indices_to_remove)]
        
    
    return train_train_indices, train_test_indices, train_indices, clipped_test_indices




#define function that will be parallelized
def parallel_encoder(sub_count,test_percent):
#for sub_count in range(1):
#can run this as a loop outside of the function for debugging
    
    train_train_indices, train_test_indices, train_indices, test_indices  = test_train_splitter(run_lens,test_percent)
    # disable warning logs
    warnings.filterwarnings('ignore')
    
    sub_list = ['sub-01','sub-02','sub-03','sub-04','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20']
    
    # x and y sizes of feature maps to be used. Can change each flexibly
    feat_map_size = np.array([7,16])
    sub = sub_list[sub_count]
    print("Loading Data")
    load_start = time.time()

    ## lOAD THE DATA 
    # load features per frames and prf estimates per voxel
    pr_estimates = np.load(os.path.join(PRF_DIR, 'no_threshold_prf_estimates_array_cover10_'+sub+'.npy'), allow_pickle=True)
    sub_features = np.load(os.path.join(HYPERLAYERS_DIR, 'Model_features_'+sub+'_hyperlayer_7_16.npy'), allow_pickle=True)
    
    # load brain data
    sub_movie_data = np.load(os.path.join(FMRI_DIR, 'normalized_masked_concat_movie_'+sub+'.npy'), allow_pickle=True)

    load_end = time.time()
    print("Data Loaded in ",load_end - load_start,' seconds')
    # load concatenated fixation frames and coordinates, and collect some info about runs
    concat_fixations = np.zeros((sub_features.shape[0],3))
    run_fix_lens = []
    run_sub_frames = []
    counter = 0 
    
    for run in ['1','2','3','4','5','6','7','8']:
        run_fixations = np.load(os.path.join(FIXATIONS_DIR, sub+'-run'+run+'.npy'), allow_pickle=True)
        concat_fixations[counter:counter+run_fixations.shape[0]] = run_fixations
        run_fix_lens.append(run_fixations.shape[0])
        run_sub_frames.append(run_fixations[:,0].astype(int))
        counter += run_fixations.shape[0]
        

    output_summary = []
    all_voxels_all_nan_fixations = []
    all_voxels_empty_trs = []

    # create model weight logs
    encoder_model_brain_weights = np.zeros((pr_estimates.shape[0],sub_features.shape[1])) #FIND OUT
    
    # select an hrf delay in frames (25frames/1sec)
    hrf_frame_offset = 112

    # Save fixation to tr placement indices
    empty_trs = []
    all_runs_tr_frames = []
    for run,run_len,run_fix_len,sub_needed_frames in zip(['1','2','3','4','5','6','7','8'],[451,441,438,488,462,439,542,338],run_fix_lens,run_sub_frames):

        sub_needed_frames_hrf = sub_needed_frames + hrf_frame_offset
        run_tr_frames = []
        #for each tr in run
        for num_tr in range(run_len):
            # get the frame numbers in tr
            tr_frames = np.where(np.floor(sub_needed_frames_hrf/50).astype(int)==num_tr)[0]
            run_tr_frames.append(tr_frames)

        all_runs_tr_frames.append(run_tr_frames)
    
    # for every voxel in the dataset
    for voxel_count, voxel in tqdm(enumerate(pr_estimates)):
        loop_start_time = time.time()
        #get prf estimates of the voxel
        fine_x = voxel[0,1] * one_vis_degree_pixel
        #flip y because pixels increase downwards
        fine_y = -voxel[1,1] * one_vis_degree_pixel
        

        # get fixations at all frames
        concat_fixations_x = concat_fixations[:,1]
        concat_fixations_y = concat_fixations[:,2]
        # add voxel prf in pixels to all fixations
        concat_fixations_x = concat_fixations_x + fine_x
        concat_fixations_y = concat_fixations_y + fine_y
        
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
        
        feature_grab_time = time.time() - loop_start_time
        #set the nan indices to 0
        voxel_feature_timeseries[all_nan_indices] = 0
        # select an hrf delay in frames (25frames/1sec)
        hrf_frame_offset = 112
        temporal_downsample_start = time.time()
        voxel_feature_timeseries_tr = np.zeros((3599,sub_features.shape[1]))
        fix_count = 0
        run_tr_count = 0
        for run_len,run_fix_len,run_tr_frames in zip([451,441,438,488,462,439,542,338],run_fix_lens,all_runs_tr_frames):

            run_voxel_feature_timeseries = voxel_feature_timeseries[fix_count:fix_count+run_fix_len]
            for num_tr,tr_frames in enumerate(run_tr_frames):
                if tr_frames.shape[0]>0: # if tr has frames, average them to get 1 tr
                    tr_average = np.nanmean(run_voxel_feature_timeseries[tr_frames],axis=0)

                else: # if not, set them to 0 or mean, decided on mean, could change later
                    # identify and save empty tr's for future check, requires manual separation of runs!
                    filler = np.empty((run_voxel_feature_timeseries.shape[1]))
                    filler[:] = np.mean(run_voxel_feature_timeseries,axis=0)
                    tr_average = filler

                # place the tr in big array
                voxel_feature_timeseries_tr[run_tr_count+num_tr]= tr_average
            fix_count += run_fix_len
            run_tr_count += run_len
        
        # lengths of runs to normalize each run separately
        zero_run_lens = [0,451,441,438,488,462,439,542,338]
        temporal_downsample_end = time.time()
        normalisation_time_start = time.time()
        # normalize feature data
        # Transpose the data
        transposed_data = voxel_feature_timeseries_tr.T  # shape: (n_features, n_timepoints)
        n_features, n_timepoints = transposed_data.shape

        # Calculate run boundaries once
        run_boundaries = np.cumsum(zero_run_lens)
        n_runs = len(zero_run_lens) - 1

        # Initialize output array
        normalized_transposed = np.zeros_like(transposed_data)

        # Loop through runs
        for run in range(n_runs):
            run_start = run_boundaries[run]
            run_end = run_boundaries[run+1]
            # Extract all features for this run at once
            run_data = transposed_data[:, run_start:run_end]  # shape: (n_features, run_length)
            # Calculate conditions for all features at once
            nonzero_sums = np.nansum(run_data, axis=1) != 0  # shape: (n_features,)
            non_nan_counts = run_data.shape[1] - np.sum(np.isnan(run_data), axis=1)  # shape: (n_features,)
            need_normalization = nonzero_sums & (non_nan_counts > 1)  # shape: (n_features,)
            # Apply log transform to all data
            log_data = np.log1p(run_data)  # shape: (n_features, run_length)
            # Initialize run output with original data
            norm_output = np.copy(run_data)  # shape: (n_features, run_length)
            if np.any(need_normalization):
                # Calculate means for all features at once
                # First create a masked array to handle NaNs properly
                masked_log_data = np.ma.masked_array(log_data, mask=np.isnan(log_data))
                # Calculate means only for features needing normalization
                means = np.zeros(n_features)
                means[need_normalization] = np.ma.mean(masked_log_data[need_normalization], axis=1).data
                # Center the data (broadcasting means across time dimension)
                # Only for features needing normalization
                centered_data = np.zeros_like(log_data)
                centered_data[need_normalization] = log_data[need_normalization] - means[need_normalization, np.newaxis]
                # Calculate standard deviations
                # Again using masked arrays to handle NaNs properly
                masked_centered = np.ma.masked_array(centered_data, mask=np.isnan(centered_data))
                stds = np.zeros(n_features)
                stds[need_normalization] = np.ma.std(masked_centered[need_normalization], axis=1).data
                # Create masks for zero and non-zero stds
                zero_std = (stds == 0) & need_normalization
                nonzero_std = (stds > 0) & need_normalization
                # For features with zero std, set output to mean
                if np.any(zero_std):
                    # Broadcasting means across all timepoints in the run
                    for feat_idx in np.where(zero_std)[0]:
                        norm_output[feat_idx] = means[feat_idx]
                # For features with non-zero std, apply z-scoring
                if np.any(nonzero_std):
                    for feat_idx in np.where(nonzero_std)[0]:
                        norm_output[feat_idx] = centered_data[feat_idx] / stds[feat_idx]
            # Store normalized data for this run
            normalized_transposed[:, run_start:run_end] = norm_output

        # Transpose back to original orientation
        normalized_voxel_feature_timeseries_tr = normalized_transposed.T

        normalisation_time_end = time.time()
        # print("Starting Processing")
        data_proc_time = time.time()
        #get voxel brain data
        voxel_movie_timeseries = sub_movie_data[voxel_count]

        #test train brain data
        x_te= voxel_movie_timeseries[test_indices] # your brain responses, test set
        x_full = voxel_movie_timeseries[train_indices]# your brain responses, training set
        x_tr = voxel_movie_timeseries[train_train_indices]# your brain responses, training set
        x_val = voxel_movie_timeseries[train_test_indices]# your brain responses, training set
    
        #test train features
        f_te = normalized_voxel_feature_timeseries_tr[test_indices] # your features.reshape(n_te, -1)
        f_full = normalized_voxel_feature_timeseries_tr[train_indices]# your features.reshape(n_tr, -1)
        f_tr = normalized_voxel_feature_timeseries_tr[train_train_indices]# your features.reshape(n_tr, -1)
        f_val = normalized_voxel_feature_timeseries_tr[train_test_indices]# your features.reshape(n_tr, -1)
        
        # new average feature timeseries
        average_feat_timeseries = np.empty(3599)
        average_feat_timeseries.fill(0)
        average_feat_timeseries=np.mean(normalized_voxel_feature_timeseries_tr,axis=1)
        
        try:
            if ~(np.sum(np.isnan(voxel_movie_timeseries))>0) :
                    
                    # # Linear Encoder Results
                    
                    ridge_cv = ModifiedKernelRidgeCV() #modifiedKernelRidgeCV
                    training_summary = ridge_cv.train(f_tr,f_val,f_full, x_tr,x_val,x_full) #,lambdass[sub_count])

                    model = training_summary['model']
                    predicted_test = model.predict(f_te)
                    # predicted_train = model.predict(f_tr)
                    
                    # place voxel predicted feature timeseries in predicted feature model brain
                    #predicted_feat_model_brain[voxel_count] = predicted_test
                
                    test_corr, test_p = pearsonr(predicted_test,x_te)
                    #train_corr, train_p = pearsonr(predicted_train,x_tr)
                    #print(test_corr)

                    test_mse = mean_squared_error(predicted_test,x_te)
                    training_summary['test_mse'] = test_mse
                    training_summary['test_corr'] = test_corr
                    training_summary['test_p'] = test_p

                    output_summary.append(training_summary)
                    

            else: 
                    
                whoopsie_summary = {'model':None,
                                'pearsonr':None,
                                'pearsonrp':None,
                                'test_mse':None,
                                'test_corr':None,
                                'test_p':None
                                }

                output_summary.append(training_summary)

 
        except:
            print('Empty Voxel at: '+str(voxel_count)+' '+sub)

    save_name = os.path.join(RESULTS_DIR, 'precision_results_dictionary_list_'+sub+'.pkl')
    with open(save_name, 'wb') as file:
        pickle.dump(output_summary, file)
        print(sub, 'Results Saved.')
    print(sub, 'Done!')

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, help="SLURM Array Task ID")
    args = parser.parse_args()
    print('task_id:',args.task_id)

    # Main logic
    dataset_to_use = sub_list[args.task_id]
    print(f"Task ID {args.task_id} -> Training participant: {dataset_to_use}")
    parallel_encoder(args.task_id,test_percent)


# ===== Main entry point =====
if __name__ == "__main__":
    main()

