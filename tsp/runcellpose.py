import os, time, sys
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from cellpose import utils, models, io
import matplotlib.pyplot as plt
from tsp.masks import filter_by_intensity


### Running Cellpose ###
def run_cellpose(files, 
                 pretrained, 
                 diameter, flow, cellprob, 
                 min_size, min_ave_intensity, min_total_intensity, 
                 plot, output, channels):
    
    if torch.cuda.is_available() :
        gpu = True
    else :
        gpu = False
        
    # Declare model #
    if(pretrained == 'cyto'):
        model = models.Cellpose(gpu=gpu, model_type='cyto')
    
    elif(pretrained == 'tissuenet'):
        model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/tissuenet_1.0/images/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_04_21_13_47_58.317948') # trained on tissuenet using cyto initial weights
        #model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/tissuenet_1.0/images/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_04_28_19_04_45.223116') # trained on tissuenet without initial weights
        #model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/cellpose_images/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_05_02_14_36_14.639818') # trained on cellpose dataset by Sunwoo
    
    elif(pretrained == 'cytotrain7'):
        model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/cellpose_trained_models/cellpose_residual_on_style_on_concatenation_off_training7_2023_01_18_16_58_51.772584') # trained on seven training images from K
    
    else:
        sys.exit("model not defined")
        
    ncells = []
    for item in files :

        img = io.imread(item); 
        filename = os.path.splitext(item)[0]
        if(pretrained == 'cyto'):
            masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=channels, flow_threshold=flow, cellprob_threshold=cellprob, min_size=min_size)
        else:
            masks, flows, styles        = model.eval(img, diameter=diameter, channels=channels, flow_threshold=flow, cellprob_threshold=cellprob, min_size=min_size)
            diams = diameter
        
        # Post-processing (min_size, min_intensity) #
        if min_ave_intensity>0 | min_total_intensity>0: # avoid running this if can because it is slow
            masks = filter_by_intensity(image=img, mask=masks, channels=channels, min_ave_intensity=min_ave_intensity, min_total_intensity=min_total_intensity) # intensity
        
        ncell = np.max(masks)
        ncells.append(ncell)
        
        save_path = filename

        # Save a plot of masks only
        outlines = utils.masks_to_outlines(masks)
        plt.imsave(save_path + "_masks.png", outlines, cmap='gray')        

        ## Save a csv file, one mask per row, include size, center_x, center_y         ##
        size_masks = np.unique(masks, return_counts=True)[1][1:].tolist()        


        centers=ndimage.center_of_mass(masks, labels=masks, index=list(range(1,ncell+1)))
        center_y=[i[0] for i in centers]
        center_x=[i[1] for i in centers]

        # center_x=[]; center_y=[]
        # for i in range(1,ncell+1):
        #     mask_pixel = np.where(masks == i)
        #     center_y.append((np.max(mask_pixel[0]) + np.min(mask_pixel[0])) / 2)
        #     center_x.append((np.max(mask_pixel[1]) + np.min(mask_pixel[1])) / 2)

        mask_res = pd.DataFrame([size_masks, center_x, center_y]).T
        mask_res.columns = ["size","center_x","center_y"]        
        mask_res.index = [f"Cell_{i}" for i in range(1,ncell+1)]
        mask_res.to_csv(filename + "_masks.csv", header=True, index=True, sep=',')
        
        
        # Optional output #
        
        # mask info
        if(output == True):
            # save _cp_outline to convert to roi by ImageJ
            outlines_list = utils.outlines_list(masks)
            io.outlines_to_text(save_path, outlines_list)
            
            # save .npy file
            io.masks_flows_to_seg(img,masks,flows,diams, file_names=save_path + '.npy')             
        
        # additional plots
        if(plot == True) :
            # Mask plot (outline) #
            my_dpi = 96
            outX, outY = np.nonzero(outlines)
            imgout= img.copy()
            imgout[outX, outY] = np.array([255,75,75]) # np.array([255,255,255]) white for severity analysis
            plt.figure(figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
            plt.gca().set_axis_off()
            plt.imshow(imgout)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(save_path + "_mask_outline.png", bbox_inches = 'tight', pad_inches = 0)
            plt.close('all')
            
            # Mask plot (fill) #
            fill_mask = (masks!=0)
            fillX, fillY = np.nonzero(fill_mask)
            if(channels == [0,0]):
                imgout = fill_mask
            else:
                imgout= img.copy()
                imgout[fillX, fillY] = np.array([255,255,255]) # white for masks
            plt.figure(figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
            plt.gca().set_axis_off()
            plt.imshow(imgout)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            if(channels == [0,0]):
                plt.imsave(save_path + "_mask_fill.png", imgout, cmap='gray')
            else:
                plt.savefig(save_path + "_mask_fill.png", bbox_inches = 'tight', pad_inches = 0)
            plt.close('all')
            
            # Mask(text) plot #
            # It takes such a long time, so it may be off for severity analysis #
            imgout= img.copy()
            plt.figure(figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
            plt.gca().set_axis_off()
            plt.imshow(imgout)
            for i in range(masks.max()):
                plt.text(center_x[i], center_y[i], str(i+1), dict(size=10, color='red', horizontalalignment='center', verticalalignment='center'))
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(save_path + "_mask_text.png", bbox_inches = 'tight', pad_inches = 0)
            plt.close('all')
            
            # Mask(center point) plot #
            imgout= img.copy()
            plt.figure(figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
            plt.gca().set_axis_off()
            plt.imshow(imgout)
            for i in range(masks.max()):
                plt.plot(center_x[i], center_y[i], marker='o', color='r', ls='', markersize=4)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(save_path + "_mask_point.png", bbox_inches = 'tight', pad_inches = 0)
            plt.close('all')
    
    # save cell counts to a text file
    ncells_mat = pd.DataFrame(list(zip(files,ncells)))
    ncells_mat.columns = ["File_name","Cell_count"]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    ncells_mat.to_csv("cellpose_counts_"+timestr+".txt", header=True, index=None, sep=',')



