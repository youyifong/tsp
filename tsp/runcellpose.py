import os, time, sys
import numpy as np
import pandas as pd
import torch
from cellpose import models, io
from tsp.masks import filter_by_intensity
from tsp.masks import save_stuff


### Running Cellpose ###
def run_cellpose(files, channels,
                 pretrained, 
                 diameter, flow, cellprob, 
                 normalize_100,
                 min_size, min_avg_intensity, min_total_intensity, 
                 save_outlines_only, save_additional_plots, save_roi, save_flow):
    
    if torch.cuda.is_available() :
        gpu = True
    else :
        gpu = False
        
    # make sure a folder named csv exists
    if not os.path.exists('csv'):
        os.makedirs('csv')

    # Declare model #
    if(pretrained in ['cyto']):
        model = models.Cellpose(gpu=gpu, model_type=pretrained)
    
    elif(pretrained in ['tissuenet']):
        model = models.CellposeModel(gpu=gpu, pretrained_model=None, model_type=pretrained)
    
    elif(pretrained == 'cytotrain7' or pretrained == 'c7'): # starting with cyto and trained with 7 images
        model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/shared/cellpose_trained_models/cellpose_residual_on_style_on_concatenation_off_training7_2023_01_18_16_58_51.772584') 
            
    elif(pretrained == 'c7s'): # starting with cyto and trained with shrink gt from Youyi
        model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/shared/cellpose_trained_models/cellpose_residual_on_style_on_concatenation_off_train_2023_09_12_15_55_26.655197') 
        
    elif(pretrained == 'c7s2'): # starting with c7s and trained with new gt from K
        model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/shared/cellpose_trained_models/cellpose_residual_on_style_on_concatenation_off_train_2023_09_12_21_34_27.005038') 
        
    elif(pretrained == 'c7s3'): # starting with c7s2 and trained with new gt from K
        model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/shared/cellpose_trained_models/cellpose_residual_on_style_on_concatenation_off_ROI_3_2023_12_14_14_00_27.160229') 
        
        
        # model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/tissuenet_1.0/images/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_04_21_13_47_58.317948') # trained on tissuenet using cyto initial weights
        # model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/tissuenet_1.0/images/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_04_28_19_04_45.223116') # trained on tissuenet without initial weights
        # model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/cellpose_images/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_05_02_14_36_14.639818') # trained on cellpose dataset by Sunwoo
    
    else:
        model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained) # trained on seven training images from K
        # sys.exit("model not defined")
        
    ncells = []
    for file in files :

        img = io.imread(file); 
        filename = os.path.splitext(file)[0]
        if(pretrained == 'cyto'):
            masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=channels, flow_threshold=flow, cellprob_threshold=cellprob, min_size=min_size, normalize_100=normalize_100)
        else:
            masks, flows, styles        = model.eval(img, diameter=diameter, channels=channels, flow_threshold=flow, cellprob_threshold=cellprob, min_size=min_size, normalize_100=normalize_100)
            diams = diameter
        
        # Post-processing (min_size, min_intensity) #
        if min_avg_intensity>0 or min_total_intensity>0: # avoid running this if can because it is slow
            masks = filter_by_intensity(image=img, masks=masks, channels=channels, min_avg_intensity=min_avg_intensity, min_total_intensity=min_total_intensity) # intensity
        
        ncells.append(len(np.unique(masks))-1)
        
        if save_flow: io.masks_flows_to_seg(img,masks,flows,diams, file_names=filename + '.npy')             
        
        save_stuff(masks, file, pretrained, channels, save_outlines_only, save_additional_plots, save_roi, img=img)

    # save cell counts to a text file
    ncells_mat = pd.DataFrame(list(zip(files,ncells)))
    ncells_mat.columns = ["File_name","Cell_count"]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    ncells_mat.to_csv("csv/cellpose_counts_"+timestr+".txt", header=True, index=None, sep=',')



