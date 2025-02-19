import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')
class modifiedKernelRidgeCV:

    def train(self, f_train, f_val, f_full, x_train, x_val, x_full, alphas):
        best_model, best_error = None, np.inf
        
        training_summary = np.zeros((5,3))
        #consider nan
        # CV over all the lambdas
        for n_lambda, lambda_, in enumerate(alphas):
            
            # k-Ridge, where regularization?
            # precomputed, then treat X as kernel (otherwise as X)
            kernel_ridge = Ridge(alpha=lambda_)
            kernel_ridge.fit(f_train, x_train)
            y = kernel_ridge.predict(f_val)
            #error = 1 - (pearsonr(x_test,y)[0])
            training_summary[n_lambda,0],training_summary[n_lambda,1] = pearsonr(x_val,y)
            training_summary[n_lambda,2] = kernel_ridge.alpha
            error = 1 - training_summary[n_lambda,0]
            best_alpha = kernel_ridge.alpha
            #error = np.sum(((self.target - y) / (1 - df_ / self.kernel.shape[0])) ** 2)
            if error < best_error:
                best_error = error
                best_model = kernel_ridge
                best_alpha = best_model.alpha
        best_model = Ridge(alpha=best_alpha)
        best_model.fit(f_full, x_full)
        
        #print("Best error:", best_error, "Alpha: ", best_model.alpha)
        return best_model, training_summary
    
def test_train_splitter(run_lens,test_percent):
    # Given a list with number of timepoints each run and desired percentage of test data,
    # Returns indices for train-test split and train_train-train_test split for model validation
    # to be used on concatenated data
    
    half_test_percent = int((test_percent/2))
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
            #print('Run Test Size: ', indices[np.cumsum(run_lens_zero)[counter-1]+lower_value:np.cumsum(run_lens_zero)[counter-1]+upper_value].shape[0])
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
            #print('Train Test Size: ', indices[np.cumsum(zero_run_train_lens)[counter-1]+lower_value:np.cumsum(zero_run_train_lens)[counter-1]+upper_value].shape[0])

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
        
    #return train_indices, clipped_test_indices, train_train_indices, train_test_indices
    #return train_indices, clipped_test_indices, train_train_indices, train_test_indices
    
    return train_train_indices, train_test_indices, train_indices, clipped_test_indices

#sub_list = ['sub-01','sub-02','sub-03','sub-04','sub-05','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20']

run_lens = [451,441,438,488,462,439,542,338]

all_subs_voxel_prf_feature_map_coordinates = np.zeros((7,14,19629,2))

# feat_map_sizes = np.array([[17,40],[17,40],[17,40],[17,40],[17,40],[17,40],[17,40]])

one_vis_degree_pixel = 1024/20

# create the parallelization here
test_percent = int(20)
#train_indices, test_indices, train_train_indices, train_test_indices = test_train_splitter(run_lens,test_run)


# load lambdas to try
lambdass = np.load('/path/Precision/Alphas_Range_vgg19/all_subs_hyperlayer_vgg19_model_alphas.npy',allow_pickle=True)
#removed sub5 so deleted sub5 alphas
lambdass = np.delete(lambdass,4,axis=1)

#define function that will be parallelized
def parallel_encoder(sub_count,lambdass,test_percent):
#for sub_count in range(1):
#can run this as a loop outside of the function for debugging
    
    train_train_indices, train_test_indices, train_indices, test_indices  = test_train_splitter(run_lens,test_percent)
    # disable warning logs
    warnings.filterwarnings('ignore')
    
    #load list of subject names, notice the missing ones, labdass should not include missing subjects
    sub_list = ['sub-01','sub-02','sub-03','sub-04','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20']
    
    # x and y sizes of feature maps to be used. Can change each flexibly
    feat_map_size = np.array([7,16])
    sub = sub_list[sub_count]
    
    # load features per frames and prf estimates per voxel
    pr_estimates = np.load('/path/pRF_Estimates/no_threshold_prf_estimates_array_cover10_'+sub+'.npy',allow_pickle=True)
    #sub_features = np.load('/path/vgg19/FeaturesBySubsRemodnav/Model_Features_'+sub+'_concat_run_layer'+str(layer_count)+'_7_16.npy')
    sub_features = np.load('/path/HyperlayersBySubsRemodnav/Model_features_'+sub+'_hyperlayer_7_16.npy',allow_pickle=True)

    # load brain data
    sub_movie_data = np.load('/path/Movie_Data/normalized_masked_concat_movie_'+sub+'.npy',allow_pickle=True)

    print('Data Loaded.')
    # load concatenated fixation frames and coordinates, and collect some info about runs
    concat_fixations = np.zeros((sub_features.shape[0],3))
    run_fix_lens = []
    run_sub_frames = []
    counter = 0 
    
    for run in ['1','2','3','4','5','6','7','8']:
        run_fixations = np.load('/path/Frames_Per_Sub_Remodnav/Needed_frames_'+sub+'-run'+run+'.npy',allow_pickle=True)
        concat_fixations[counter:counter+run_fixations.shape[0]] = run_fixations
        run_fix_lens.append(run_fixations.shape[0])
        run_sub_frames.append(run_fixations[:,0].astype(int))
        counter += run_fixations.shape[0]
        
    # create empty average feature model brain
    average_feat_model_brain = np.zeros((pr_estimates.shape[0],3599))
    # create empty predicted feature model brain
    predicted_feat_model_brain = np.zeros((pr_estimates.shape[0],test_indices.shape[0]))
    # create empty result log array
    output_summary = np.zeros((pr_estimates.shape[0],14,3))
    # create NAN FIXATION logs and EMPTY TRS logs
    all_voxels_all_nan_fixations = []
    all_voxels_empty_trs = []

    # create model weight logs
    encoder_model_brain_weights = np.zeros((pr_estimates.shape[0],sub_features.shape[1])) #FIND OUT
    
    print('Starting Analysis Loop...')
    # for every voxel in the dataset
    for voxel_count, voxel in enumerate(pr_estimates):
        #get prf estimates of the voxel
        fine_x = voxel[0,1] * one_vis_degree_pixel
        #flip y because pixels increase downwards
        fine_y = -voxel[1,1] * one_vis_degree_pixel
        
        # get  !!!!!FLIPPED!!!!!! fixations at all frames
        concat_fixations_x = np.flip(concat_fixations[:,1],0)
        concat_fixations_y = np.flip(concat_fixations[:,2],0)
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
        
        #set the nan indices to 0
        voxel_feature_timeseries[all_nan_indices] = 0
        # select an hrf delay in frames (25frames/1sec)
        hrf_frame_offset = 112
        
        #downsample voxel feature timeseries to TR's 
        voxel_feature_timeseries_tr = np.zeros((3599,sub_features.shape[1]))
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
        
        # place voxel average feature timeseries in average feature model brain
        average_feat_model_brain[voxel_count] = average_feat_timeseries
        

        # if (output_summary[voxel_count,10,0]/655 < 0.5) |( output_summary[voxel_count,11,0]/2880 < 0.5) | (output_summary[voxel_count,12,0]/574 < 0.5) | (output_summary[voxel_count,13,0]/2306 < 0.5) |  (np.sum(np.isnan(voxel_movie_timeseries))>0):
        if ~(np.sum(np.isnan(voxel_movie_timeseries))>0) & ~(np.sum(np.isnan(normalized_voxel_feature_timeseries_tr))>0) & ~np.all(normalized_voxel_feature_timeseries_tr==0) :
            try:
            # # Full timeseries correlation
                av_feat_corr, av_feat_p = pearsonr(np.nanmean(normalized_voxel_feature_timeseries_tr,axis=1),voxel_movie_timeseries)
            
                output_summary[voxel_count,0,0] = av_feat_corr
                output_summary[voxel_count,0,1] = av_feat_p
            
            # # Linear Encoder Results
            
                ridge_cv = modifiedKernelRidgeCV()
                model,training_summary = ridge_cv.train(f_tr,f_val,f_full, x_tr,x_val,x_full,lambdass[sub_count])
    
                predicted_test = model.predict(f_te)
                predicted_train = model.predict(f_tr)
            
            # place voxel predicted feature timeseries in predicted feature model brain
                predicted_feat_model_brain[voxel_count] = predicted_test
        
                test_corr, test_p = pearsonr(predicted_test,x_te)
                train_corr, train_p = pearsonr(predicted_train,x_tr)
     
                output_summary[voxel_count,3,0] = test_corr
                output_summary[voxel_count,3,1] = test_p
            
                output_summary[voxel_count,4,0] = train_corr
                output_summary[voxel_count,4,1] = train_p
            
                output_summary[voxel_count,5:10] = training_summary
            
                model_weights = model.coef_
                encoder_model_brain_weights[voxel_count] = model_weights
            except:
                print('Error in '+sub+ 'Voxel '+ str(voxel_count))
        else: 
            
            output_summary[voxel_count,0,0] = 0
            output_summary[voxel_count,0,1] = 1
            
            output_summary[voxel_count,3,0] = 0
            output_summary[voxel_count,3,1] = 1
            
            output_summary[voxel_count,4,0] = 0
            output_summary[voxel_count,4,1] = 1
            
            output_summary[voxel_count,5:10] = 999
            
            encoder_model_brain_weights[voxel_count] = 999
            print(voxel_count,flush=True)
    print('Saving outputs...')
    # save outputs
    save_title = sub.upper()+'_HyperLayer_HRF_Delay_'+str(np.round(hrf_frame_offset/25,decimals=1)).replace('.','_')+'_Seconds'
    
    np.save('/path/flipped_gaze_'+save_title+'_voxel_predicted_response_7_16',predicted_feat_model_brain,allow_pickle=True)
    np.save('/path/flipped_gaze_'+save_title+'_voxel_model_weights_7_16',encoder_model_brain_weights,allow_pickle=True)

    np.save('/path/flipped_gaze_'+save_title+'_all_voxels_nan_indices_7_16',all_voxels_all_nan_fixations,allow_pickle=True)
    np.save('/path/flipped_gaze_'+save_title+'_all_voxels_empty_trs_7_16',all_voxels_empty_trs,allow_pickle=True)

    np.save('/path/flipped_gaze_'+save_title+'_encoding_output_summary_7_16',output_summary,allow_pickle=True)
    np.save('/path/flipped_gaze_'+save_title+'_average_feature_model_brain_7_16',average_feat_model_brain,allow_pickle=True)
    print(sub_count, ' Done.')
    
Parallel(n_jobs=3,prefer='threads')(delayed(parallel_encoder)(sub_count,lambdass,test_percent)for sub_count in [0,1,2,3,4,5,6,7,8,9,10,11,12])#,2,3,4,5,6,7,8,9,10,11,12])
