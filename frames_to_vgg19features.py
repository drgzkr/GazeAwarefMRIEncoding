import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os
from tqdm import trange
from joblib import Parallel, delayed
from config import FRAMES_DIR, FIXATIONS_DIR, VGG_BY_RUN_DIR

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)

model.eval()

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

shrink_features = transforms.Compose([
    transforms.Resize((7,16)),
    
])
        
def save_features(runs):
    save_dir = VGG_BY_RUN_DIR
    
    dir_name = os.path.join(FRAMES_DIR, 'Run'+str(runs))
    only_frames = np.load(os.path.join(FIXATIONS_DIR, 'Needed_frames_Run_'+str(runs)+'.npy')) 

    for layer_count, layer in enumerate([7,14,27,40,53]):
        
        data =[]
        
        fname = dir_name + 'frame'+str(only_frames[0])+'.jpg'

            
        input_image = Image.open(fname)
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')
    
        fname = dir_name + 'frame'+str(only_frames[0])+'.jpg'
        #  pre-process the image
        input_image = Image.open(fname)
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        # Run the predictions
        layer1 = model.features[:layer](input_batch)
        layer1_shrunk = shrink_features(layer1)
        
        #layer1_array = layer1.detach().numpy()
        layer1_shrunk_array = layer1_shrunk.detach().numpy()
        
        
        data_shape = list(layer1_shrunk_array.shape)
        data_shape[0] = np.shape(only_frames)[0]
        data_shape = tuple(data_shape)
        data = np.zeros(data_shape)
        

        for j in trange(len(only_frames)): 
    
            fname = dir_name + 'frame'+str(only_frames[j])+'.jpg'
            if os.path.isfile(fname):
                #  pre-process the image
                input_image = Image.open(fname)
                input_tensor = preprocess(input_image)
                input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
                
                # move the input and model to GPU for speed if available
                if torch.cuda.is_available():
                    input_batch = input_batch.to('cuda')
                    model.to('cuda')
                # Run the predictions
                layer1 = model.features[:layer](input_batch)
                layer1_shrunk = shrink_features(layer1)
                layer1_shrunk_array = layer1_shrunk.detach().numpy()
                # Save them to data array
                data[j]= layer1_shrunk_array[0]
            

        np.save(os.path.join(save_dir, 'Run'+str(runs)+'_Layer'+str(layer_count)+'_vgg19_features_resized_to_7_16.npy'), data)

    
    
Parallel(n_jobs=4,verbose=10)(delayed(save_features)(runs)for runs in [1,2,3,4,5,6,7,8])  
    
