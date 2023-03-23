import os, time
import numpy as np
import pandas as pd
import torch
from cellpose import utils, models, io
import matplotlib.pyplot as plt

from tsp.masks import fill_holes_and_remove_small_masks, GetCenterCoor, Intensity


### Running Cellpose ###
def run_cellpose(files, 
                 pretrained, 
                 diameter, flow, cellprob, 
                 minsize, min_ave_intensity, min_total_intensity, 
                 plot, output, channels):
    
    if torch.cuda.is_available() :
        gpu = True
    else :
        gpu = False
        
    # Declare model #
    if(pretrained == 'cyto'):
        model = models.Cellpose(gpu=gpu, model_type='cyto')
    
    if(pretrained == 'tissuenet'):
        model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/tissuenet_1.0/images/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_04_21_13_47_58.317948') # trained on tissuenet using cyto initial weights
        #model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/tissuenet_1.0/images/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_04_28_19_04_45.223116') # trained on tissuenet without initial weights
        #model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/cellpose_images/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_05_02_14_36_14.639818') # trained on cellpose dataset by Sunwoo
    
    if(pretrained == 'cytotrain7'):
        model = models.CellposeModel(gpu=gpu, pretrained_model='/fh/fast/fong_y/cellpose_trained_models/cellpose_residual_on_style_on_concatenation_off_training7_2023_01_18_16_58_51.772584') # trained on seven training images from K
    
    ncell = []
    for item in files :
        img = io.imread(item); 
        filename = os.path.splitext(item)[0]
        if(pretrained == 'cyto'):
            masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=channels, flow_threshold=flow, cellprob_threshold=cellprob)
        else:
            masks, flows, styles = model.eval(img, diameter=diameter, channels=channels, flow_threshold=flow, cellprob_threshold=cellprob)
            diams = diameter
        
        # Post-processing (min_size, min_intensity) #
        masks = fill_holes_and_remove_small_masks(masks, min_size=minsize) # minsize
        res_intensity = Intensity(image=img, mask=masks, channels=channels, min_ave_intensity=min_ave_intensity, min_total_intensity=min_total_intensity) # intensity
        masks = res_intensity.mask
        
        ncell.append((len(np.unique(masks))-1))
        
        save_path = filename

        # Save masks info #        
        if(output == True):
            # save _cp_outline to convert to roi by ImageJ
            outlines = utils.outlines_list(masks)
            io.outlines_to_text(save_path, outlines)
            
            # save .npy file
            io.masks_flows_to_seg(img,masks,flows,diams, file_names=save_path + '.npy') 
            
            # Size #
            size_masks = []
            act_mask = np.delete(np.unique(masks),0)
            for i in act_mask:
                mask_pixel = np.where(masks == i)
                size_masks.append(len(mask_pixel[0]))
            # XY coordinates #
            outlines = GetCenterCoor(masks)
            mask_res = pd.DataFrame([size_masks, [i[0] for i in outlines], [i[1] for i in outlines]]).T
            mask_res.columns = ["size","center_x","center_y"]
            cellnames = []
            for i in range(mask_res.shape[0]): cellnames.append("Cell_" + str(i+1))
            mask_res.index = cellnames
            mask_res.to_csv(filename + "_sizes_coordinates.csv", header=True, index=True, sep=',')
        
        # Save plot #

        # always save a plot of masks only
        outlines = utils.masks_to_outlines(masks)
        plt.imsave(save_path + "_masks.png", outlines, cmap='gray')
        
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
            yx_center = GetCenterCoor(masks)
            y_coor = list(zip(*yx_center))[0]
            x_coor = list(zip(*yx_center))[1]
            imgout= img.copy()
            plt.figure(figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
            plt.gca().set_axis_off()
            plt.imshow(imgout)
            for i in range(masks.max()):
                plt.text(y_coor[i], x_coor[i], str(i+1), dict(size=10, color='red', horizontalalignment='center', verticalalignment='center'))
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
                plt.plot(y_coor[i], x_coor[i], marker='o', color='r', ls='', markersize=4)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(save_path + "_mask_point.png", bbox_inches = 'tight', pad_inches = 0)
            plt.close('all')
    
    ncell_mat = pd.DataFrame(list(zip(files,ncell)))
    ncell_mat.columns = ["File_name","Cell_count"]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    ncell_mat.to_csv("cellpose_counts_"+timestr+".txt", header=True, index=None, sep=',')



