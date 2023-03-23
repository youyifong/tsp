import os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cellpose import utils
from tsp.masks import GetCenterCoor
from tsp import imread

def StainingAnalysis(files, marker_names, positives, cutoffs, channels, methods, plot, output):
    
    plus_minus = ['+' if positives[l] else '-' for l in range(len(marker_names))]
    filenames=[os.path.splitext(f)[0] for f in files]
    staged_output_file_names = [filenames[0]]
    tmp =  [marker_names[i] + plus_minus[i] + str(cutoffs[i]) for i in range(len(marker_names))]
    staged_output_file_names = staged_output_file_names + [filenames[0]+"_"+"".join(tmp[:(i+1)]) for i in range(len(marker_names))]
    output_file_name = staged_output_file_names[-1]
    
    image_base = imread(files[0])      
    
    pos_rate = []; num_cell = []; mask_idx = []; masks = []
    for i in range(len(files)-1):
        positive=positives[i]
        cutoff=cutoffs[i]
        method=methods[i]

        if(i == 0):
            datA = np.load(os.path.splitext(files[0])[0] + '_seg.npy', allow_pickle=True).item()
            maskA = datA['masks']
            masks.append(maskA)
            num_cell.append(maskA.max())
            
        # Method (Positivity or Intensity) #
        if(method == 'Mask'):
            datB = np.load(os.path.splitext(files[i+1])[0] + '_seg.npy', allow_pickle=True).item()
            maskB = datB['masks']
        elif (method == 'Intensity_avg_pos' or method == 'Intensity_avg_all' or method == 'Intensity_total'):
            image_comp = imread(files[i+1])
        
        # Double staining #
        if(method == 'Mask'):
            pos_rate, num_double_cell, double_mask_idx = DoubleStain(maskA=maskA, maskB=maskB, positive=positive, cutoff=cutoff, channels=channels, method=method)
        elif(method == 'Intensity_avg_pos' or method == 'Intensity_avg_all' or method == 'Intensity_total'):
            pos_rate, num_double_cell, double_mask_idx = DoubleStain(maskA=maskA, maskB=image_comp, positive=positive, cutoff=cutoff, channels=channels, method=method)

        # last i
        if(i == len(files)-2):
            if(method == 'Mask'):
                cutoff_all = list(np.around(np.linspace(start=0, stop=1, num=11),1)) # [0,0.1,...,0.9,1]                
            elif(method == 'Intensity_avg_pos' or method == 'Intensity_avg_all' or method == 'Intensity_total'):
                cutoff_all = list(np.around(np.quantile(pos_rate, np.linspace(currentwd=0, stop=1, num=11)),1)) # quantile
                #cutoff_all = [0.0,0.5,0.7,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,4.9,5.0,5.5,6.0,7.0,8.0,9.0,10.0] # for severity analysis
            
            num_double_cell_cutoff = [] # number of double stained cell over all cutoffs
            for k in cutoff_all: 
                num_cell_temp = []
                for j in np.arange(0,len(pos_rate)) :
                    if positive: 
                        num_cell_temp.append(pos_rate[j] >= k)
                    else:
                        num_cell_temp.append(pos_rate[j] <= k)
                num_double_cell_cutoff.append(sum(num_cell_temp))
            ncell_res_temp = pd.DataFrame(list(zip(cutoff_all, num_double_cell_cutoff)))
            ncell_res_temp.columns = ["Cutoff", "Cell_count"]
            ncell_res_temp.to_csv(output_file_name + "_counts_lastcutoff.txt", header=True, index=None, sep=',')
        
        pos_rate.append(pos_rate)
        num_cell.append(num_double_cell)
        mask_idx.append(double_mask_idx)
        if(len(files) == 2):
            masks.append(GetMaskCutoff(mask=maskA, act_mask_idx=mask_idx[i]))
        if(len(files) > 2):
            maskA = GetMaskCutoff(mask=maskA, act_mask_idx=mask_idx[i])
            masks.append(maskA)
    
    
    # save masks to a csv file
    for i in range(len(files)-1):
        # Size
        size_masks = []
        act_mask = np.delete(np.unique(masks[i+1]),0)
        for idx in act_mask:
            mask_pixel = np.where(masks[i+1] == idx)
            size_masks.append(len(mask_pixel[0]))
        
        # XY coordinates 
        outlines = GetCenterCoor(masks[i+1])
        mask_res = pd.DataFrame([size_masks, [i[0] for i in outlines], [i[1] for i in outlines]]).T
        mask_res.columns = ["size","center_x","center_y"]
        
        cellnames = []
        for i in range(mask_res.shape[0]): cellnames.append("Cell_" + str(i+1))
        mask_res.index = cellnames
        mask_res.to_csv(output_file_name + "_masks.csv", header=True, index=True, sep=',')

    # save counts
    filenames_save = [files[0]] # first filename
    for i in range(len(files)-1):
        if positives[i]: 
            filenames_save.append("+" + files[i+1])
        else:
            filenames_save.append("-" + files[i+1])
    ncell_res = pd.DataFrame(list(zip(filenames_save, num_cell)))
    ncell_res.columns = ["File_name", "Cell_count"]
    ncell_res.to_csv(output_file_name + "_counts_multistain.txt", header=True, index=None, sep=',')


    # optional output 
    if(plot):
        mask_color = [255,250,240]
        for i in range(len(masks)):
            PlotMask_outline(mask=masks[i], image=files[i], filename=staged_output_file_names[i], positive=positive, color=mask_color)
            PlotMask_fill(mask=masks[i], image=files[i], filename=staged_output_file_names[i], positive=positive)
            PlotCenter(mask=masks[i], image=files[i], filename=staged_output_file_names[i], positive=positive, color='r')
    
    if(output):
        for i in range(len(files)-1):
            np.savez(file=output_file_name + '_seg', img=image_base, masks=masks[i+1])
            


# Utilites for double staining analysis
def DoubleStain(maskA, maskB, positive, cutoff, channels, method):
    # Pre-processing #
    if(method == 'Intensity_avg_pos' or method == 'Intensity_avg_all' or method == 'Intensity_total'):
        if(channels != [0,0]):
            maskB = maskB[:,:,(channels[0]-1)]
            maskB = maskB * (99/255) # normalization
        if(channels == [0,0]):
            maskB = maskB * (99/65535) # maximum value of pixels for 16 bit grayscale iamge is 65535
    
    # Double staining #
    act_idx = np.unique(maskA)
    if(sum(act_idx==0) != 0) : act_idx = np.delete(act_idx,0) # select masks only (remove 0)
    res = [] # positivity rate or intensity
    if(method == 'Mask'):
        for i in act_idx :
            cell = np.where(maskA == i)
            Bmasks = []
            for j in range(len(cell[0])) :
                temp = maskB[cell[0][j], cell[1][j]]
                if(temp != 0): Bmasks.append(temp)
            if Bmasks != []:
                temp = np.histogram(Bmasks, bins=np.append(np.unique(Bmasks), np.inf))    
                res.append( np.max(temp[0]) / len(cell[0]) )
            else: 
                res.append(0.0)
    if(method == 'Intensity_avg_pos' or method == 'Intensity_avg_all' or method == 'Intensity_total'):
        for i in act_idx:
            intensity_temp = []
            cell = np.where(maskA == i)
            for j in range(len(cell[0])) :
                temp = maskB[cell[0][j], cell[1][j]]
                intensity_temp.append(temp)
            if(method == 'Intensity_total'):
                res.append(sum(intensity_temp)) # total intensity
            if(method == 'Intensity_avg_pos'):
                intensity_temp_arr = np.array(intensity_temp)
                int_norm_avg_pos = sum(intensity_temp_arr[intensity_temp_arr != 0]) / sum(intensity_temp_arr != 0)
                if(math.isnan(int_norm_avg_pos)):
                    res.append(0) # average intensities of positive pixels after normalization
                else:
                    res.append(int_norm_avg_pos) # average intensities of positive pixels after normalization
            if(method == 'Intensity_avg_all'):
                res.append(np.mean(intensity_temp)) # average intensities of all pixels after normalization
    
    act_mask_idx = [] # double stained mask index
    for j in range(len(res)) :
        if(positive == True): act_mask_idx.append(res[j] >= cutoff)
        if(positive == False): act_mask_idx.append(res[j] <= cutoff)
    double_mask_idx = act_idx[act_mask_idx]
    if(positive == True) : num_double_stain = [l for l in res if l >= cutoff] # number of double stained cells
    if(positive == False) : num_double_stain = [l for l in res if l <= cutoff] # number of double stained cells
    return res, len(num_double_stain), double_mask_idx

def GetMaskCutoff(mask, act_mask_idx):
    act_idx = act_mask_idx
    total_idx = np.unique(mask)
    inact_idx = np.array(list(set(total_idx) - set(act_idx))) #inact_idx = np.delete(total_idx, mask_idx)
    mask_cutoff = mask.copy()
    for i in np.arange(0,len(inact_idx)) :
        mask_cutoff[mask_cutoff==inact_idx[i]] = 0
    return mask_cutoff

def PlotMask_outline(mask, image, filename, positive, color):
    # Plotting #
    img = imread(image)
    my_dpi = 96
    outlines_temp = utils.masks_to_outlines(mask)
    outX_temp, outY_temp = np.nonzero(outlines_temp)
    if(img.ndim == 3):
        imgout= img.copy()
        imgout[outX_temp, outY_temp] = np.array(color)
    if(img.ndim == 2):
        imgout = outlines_temp
    plt.figure(figsize=(mask.shape[1]/my_dpi, mask.shape[0]/my_dpi), dpi=my_dpi)
    plt.gca().set_axis_off()
    plt.imshow(imgout)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if(img.ndim == 2):
        plt.imsave(filename + '_outline.png', imgout, cmap='gray')
    if(img.ndim == 3):
        plt.savefig(filename + '_outline.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close('all')

def PlotMask_fill(mask, image, filename, positive):
    
    # Plotting #
    img = imread(image)
    my_dpi = 96
    fill_temp = (mask!=0)
    fillX_temp, fillY_temp = np.nonzero(fill_temp)
    if(img.ndim == 3):
        imgout= img.copy()
        imgout[fillX_temp, fillY_temp] = np.array([255,255,255]) # white
    if(img.ndim == 2):
        imgout = fill_temp
    plt.figure(figsize=(mask.shape[1]/my_dpi, mask.shape[0]/my_dpi), dpi=my_dpi)
    plt.gca().set_axis_off()
    plt.imshow(imgout)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if(img.ndim == 2):
        plt.imsave(filename + '_fill.png', imgout, cmap='gray')
    if(img.ndim == 3):
        plt.savefig(filename + '_fill.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close('all')

def PlotCenter(mask, image, filename, positive, color):
    # Plotting #
    yx_center = GetCenterCoor(mask)
    y_coor = list(zip(*yx_center))[0]
    x_coor = list(zip(*yx_center))[1]
    img = imread(image)
    my_dpi = 96
    imgout = img.copy()
    plt.figure(figsize=(mask.shape[1]/my_dpi, mask.shape[0]/my_dpi), dpi=my_dpi)
    plt.gca().set_axis_off()
    plt.imshow(imgout)
    for i in range(len(np.unique(mask))-1):
        plt.plot(y_coor[i], x_coor[i], marker='o', color=color, ls='', markersize=4)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if(img.ndim == 2):
        plt.imsave(filename + '_point.png', imgout, cmap='gray')
    if(img.ndim == 3):
        plt.savefig(filename + '_point.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close('all')


