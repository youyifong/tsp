# Libraries
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cellpose import utils, io
import argparse
import sys

# Arguments
ap = argparse.ArgumentParser()

args = ap.parse_args()
if(args.i == 'False'): intensity = False
if(args.i == 'True'): intensity = True
if(args.r == 'False'): output = False
if(args.r == 'True'): output = True
#print(args)

# Utilites for double staining analysis
class DoubleStain:
    def __init__(self, maskA, maskB, positive, cutoff, channels, method):
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
        self.pos_rate = res
        self.num_double_cell = len(num_double_stain)
        self.double_mask_idx = double_mask_idx

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
    img = io.imread(image)
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
    img = io.imread(image)
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
    img = io.imread(image)
    my_dpi = 96
    imgout = img.copy()
    plt.figure(figsize=(mask.shape[1]/my_dpi, mask.shape[0]/my_dpi), dpi=my_dpi)
    plt.gca().set_axis_off()
    plt.imshow(imgout)
    for i in range(len(np.unique(mask))-1):
        plt.plot(y_coor[i], x_coor[i], marker='o', color=color, ls='', markersize=2)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if(img.ndim == 2):
        plt.imsave(filename + '_point.png', imgout, cmap='gray')
    if(img.ndim == 3):
        plt.savefig(filename + '_point.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close('all')


class StainingAnalysis:
    def __init__(self, files, marker_names, positive, cutoff, channels, method, plot=True, output=output):
        
        image_base = io.imread(files[0])        
        
        pos_rate = []; num_cell = []; mask_idx = []; masks = []
        for i in range(len(files)-1):
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
                image_comp_path = [os.path.relpath(files[i+1], currentwd)]
                image_comp = io.imread(image_comp_path[-1])
            
            if(i == (len(files)-2)):
                # last i
                # Double staining #
                if(method == 'Mask'):
                    res = DoubleStain(maskA=maskA, maskB=maskB, positive=positive[i], cutoff=cutoff[i], channels=channels, method=method)
                    cutoff_all = list(np.around(np.linspace(start=0, stop=1, num=11),1)) # [0,0.1,...,0.9,1]                
                elif(method == 'Intensity_avg_pos' or method == 'Intensity_avg_all' or method == 'Intensity_total'):
                    res = DoubleStain(maskA=maskA, maskB=image_comp, positive=positive[i], cutoff=cutoff[i], channels=channels, method=method)
                    cutoff_all = list(np.around(np.quantile(res.pos_rate, np.linspace(currentwd=0, stop=1, num=11)),1)) # quantile
                    #cutoff_all = [0.0,0.5,0.7,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,4.9,5.0,5.5,6.0,7.0,8.0,9.0,10.0] # for severity analysis
                
                num_double_cell_cutoff = [] # number of double stained cell over all cutoffs
                for k in cutoff_all: 
                    num_cell_temp = []
                    for j in np.arange(0,len(res.pos_rate)) :
                        if(positive[i] == True): num_cell_temp.append(res.pos_rate[j] >= k)
                        if(positive[i] == False): num_cell_temp.append(res.pos_rate[j] <= k)
                    num_double_cell_cutoff.append(sum(num_cell_temp))
                ncell_res_temp = pd.DataFrame(list(zip(cutoff_all, num_double_cell_cutoff)))
                ncell_res_temp.columns = ["Cutoff", "Cell_count"]
                ncell_res_temp.to_csv(output_file_name + "_cutoff_counts.txt", header=True, index=None, sep=',')
            else:
                if(method == 'Mask'):
                    res = DoubleStain(maskA=maskA, maskB=maskB, positive=positive[i], cutoff=cutoff[i], channels=channels, method=method)
                elif(method == 'Intensity_avg_pos' or method == 'Intensity_avg_all' or method == 'Intensity_total'):
                    res = DoubleStain(maskA=maskA, maskB=image_comp, positive=positive[i], cutoff=cutoff[i], channels=channels, method=method)
            
            pos_rate.append(res.pos_rate)
            num_cell.append(res.num_double_cell)
            mask_idx.append(res.double_mask_idx)
            if(len(files) == 2):
                masks.append(GetMaskCutoff(mask=maskA, act_mask_idx=mask_idx[i]))
            if(len(files) > 2):
                maskA = GetMaskCutoff(mask=maskA, act_mask_idx=mask_idx[i])
                masks.append(maskA)
        self.pos_rate = pos_rate
        self.num_cell = num_cell
        self.mask_idx = mask_idx
        self.masks = masks
        
        # Plotting #
        if(plot == True):
            mask_color = [255,250,240]
            for i in range(len(masks)):
                PlotMask_outline(mask=masks[i], image=files[i], filename=staged_output_file_names[i], positive=positive, color=mask_color)
                PlotMask_fill(mask=masks[i], image=files[i], filename=staged_output_file_names[i], positive=positive)
                PlotCenter(mask=masks[i], image=files[i], filename=staged_output_file_names[i], positive=positive, color='r')
        
        # Save output #
        if(output == True):
            for i in range(len(files)-1):
                np.savez(file=output_file_name + '_seg', img=image_base, masks=masks[i+1])
                
                # Size
                size_masks = []
                act_mask = np.delete(np.unique(masks[i+1]),0)
                for idx in act_mask:
                    mask_pixel = np.where(masks[i+1] == idx)
                    size_masks.append(len(mask_pixel[0]))
                
                # XY coordinates 
                outlines = GetCenterCoor(masks[i+1])
                mask_res = pd.DataFrame([size_masks,outlines]).T
                mask_res.columns = ["size","xy_coordinate"]
                cellnames = []
                for i in range(mask_res.shape[0]): cellnames.append("Cell_" + str(i+1))
                mask_res.index = cellnames
                mask_res.to_csv(output_file_name + "_sizes_coordinates.txt", header=True, index=True, sep=',')
        
        filenames_save = [files[0]] # first filename
        for i in range(len(files)-1):
            if(positive[i] == True): filenames_save.append("+" + files[i+1])
            if(positive[i] == False): filenames_save.append("-" + files[i+1])
        ncell_res = pd.DataFrame(list(zip(filenames_save, num_cell)))
        ncell_res.columns = ["File_name", "Cell_count"]
        ncell_res.to_csv(output_file_name + "_multistain_counts.txt", header=True, index=None, sep=',')






# Utilites for intensity analysis
class MeasureIntensity:
    def __init__(self, mask, image, channels):
        if(channels != [0,0]): image = image[:,:,(channels[0]-1)]
        
        if(channels != [0,0]): image_norm = image * (99/255) # normalization for RGB image
        if(channels == [0,0]): image_norm = image * (99/65535) # normalization for grayscale image
               
        act_idx = np.unique(mask)
        if(sum(act_idx==0) != 0): act_idx = np.delete(act_idx,0) # select masks only (remove 0)
        intensity = []; intensity_norm_avg_all = []; intensity_norm_avg_pos = []; intensity_norm_total = []
        for j in act_idx :
            mask_pixel = np.where(mask == j) # mask pixels
            pixel_int = []; pixel_norm_int = []
            for k in range(len(mask_pixel[0])):
                pixel_int.append(image[mask_pixel[0][k], mask_pixel[1][k]])
                pixel_norm_int.append(image_norm[mask_pixel[0][k], mask_pixel[1][k]])
            intensity.append(sum(pixel_int)) # total intensities
            intensity_norm_total.append(sum(pixel_norm_int)) # total intensities after normalization
            intensity_norm_avg_all.append(np.mean(pixel_norm_int)) # average intensities of all pixels after normalization
            pixel_norm_int_arr = np.array(pixel_norm_int)
            int_norm_avg_pos = sum(pixel_norm_int_arr[pixel_norm_int_arr != 0]) / sum(pixel_norm_int_arr != 0)
            if(math.isnan(int_norm_avg_pos)):
                intensity_norm_avg_pos.append(0) # average intensities of positive pixels after normalization
            else:
                intensity_norm_avg_pos.append(int_norm_avg_pos) # average intensities of positive pixels after normalization
        self.intensity = np.around(intensity,1)
        self.intensity_norm_total = np.around(intensity_norm_total,1)
        self.intensity_norm_avg_all = np.around(intensity_norm_avg_all,1)
        self.intensity_norm_avg_pos = np.around(intensity_norm_avg_pos,1)

def GetCenterCoor(masks):
    outline_list = utils.outlines_list(masks)
    yx_center = []
    for mask in outline_list:
        y_coor = list(zip(*mask))[0]
        x_coor = list(zip(*mask))[1]
        y_coor_min, y_coor_max = np.min(y_coor), np.max(y_coor)
        x_coor_min, x_coor_max = np.min(x_coor), np.max(x_coor)
        y_center, x_center = (y_coor_min+y_coor_max)/2, (x_coor_min+x_coor_max)/2
        yx_center.append([y_center, x_center])
    return yx_center

class IntensityAnalysis:
    def __init__(self, files, filenames, channels):
        #image_base_path = [os.path.relpath(files[0], currentwd) + '.tiff']
        image_base_path = [os.path.relpath(files[0], currentwd)]
        image_base = io.imread(image_base_path[-1])
        if(files[0][-5:] == '.tiff'): files_temp = files[0][0:-5]
        if(files[0][-4:] == '.png'): files_temp = files[0][0:-4]
        mask_path = [os.path.relpath(files_temp, currentwd) + '_seg.npy']
        #mask_path = [os.path.relpath(files[0], currentwd) + '_seg.npy']
        dat = np.load(mask_path[0], allow_pickle=True).item()
        mask = dat['masks']
        outlines = GetCenterCoor(mask)
        
        intensity_total = []
        for i in range(len(files)):
            if(i == 0):
                res = MeasureIntensity(mask=mask, image=image_base, channels=channels)
            else:
                #image_comp_path = [os.path.relpath(files[i], currentwd) + '.tiff']
                image_comp_path = [os.path.relpath(files[i], currentwd)]
                image_comp = io.imread(image_comp_path[-1])
                res = MeasureIntensity(mask=mask, image=image_comp, channels=channels)
            #intensity_total.append(res.intensity); intensity_total.append(res.intensity_norm_ave); intensity_total.append(res.intensity_norm_total)
            intensity_total.append(res.intensity_norm_avg_all); 
            intensity_total.append(res.intensity_norm_avg_pos); 
            intensity_total.append(res.intensity_norm_total)
        intensity_total.append(list(outlines))
        intensity_res = pd.DataFrame(intensity_total).T
        colnames = []
        for i in range(len(filenames)):
            #temp = [filenames[i] + "_intensity_total", filenames[i] + "_intensity_norm_avg", filenames[i] + "_intensity_norm_total"]
            temp = [filenames[i] + "_intensity_avg_all", filenames[i] + "_intensity_avg_pos", filenames[i] + "_intensity_total"]
            for j in range(3):
               colnames.append(temp[j])
        colnames.append("xy_coordinate")
        intensity_res.columns = colnames
        cellnames = []
        for i in range(intensity_res.shape[0]): 
            cellnames.append("Cell_" + str(i+1))
        intensity_res.index = cellnames
        intensity_res.to_csv(filenames[0] + "_intensity.txt", header=True, index=True, sep=',')



# Pre-processing arguments
# File and Filename (staining and intensity analysis)
currentwd = os.getcwd()
filenames = args.f
filenames = filenames[1:-1] # remove []
filenames = filenames.split(",")
files = []
for k in range(len(filenames)) :
    total_path = currentwd + "/" + filenames[k]
    file = os.path.relpath(total_path, currentwd)
    files.append(file)
filenames = [os.path.splitext(f)[0] for f in files]


# Image channels
if(args.l == 'None'):
    channels=[3,0]
else:
    channels = args.l
    channels = channels[1:-1] # remove []
    channels = channels.split(",")
    for i in range(len(channels)): channels[i] = int(channels[i])


# Positive, Cutoff, Method (staining analysis only)
if(intensity == False):
    positive = args.p
    positive = positive[1:-1] # remove []
    positive = positive.split(",")
    positive_temp = []
    for i in range(len(positive)):
        if(positive[i] == 'True'): positive_temp.append(True)
        if(positive[i] == 'False'): positive_temp.append(False)
    positive = positive_temp
    
    cutoff = args.c
    cutoff = cutoff[1:-1] # remove []
    cutoff = cutoff.split(",")
    cutoff_temp = []
    for i in range(len(cutoff)):
        cutoff_temp.append(float(cutoff[i]))
    cutoff = cutoff_temp
    
    method = args.m
    method = method[1:-1] # remove []


marker_names = args.n[1:-1] # remove []
marker_names = marker_names.split(",")

plus_minus = ['+' if positive[l] else '-' for l in range(len(marker_names))]
staged_output_file_names = [filenames[0]]
tmp =  [marker_names[i] + plus_minus[i] + str(cutoff[i]) for i in range(len(marker_names))]
staged_output_file_names = staged_output_file_names + [filenames[0]+"_"+"".join(tmp[:(i+1)]) for i in range(len(marker_names))]
output_file_name = staged_output_file_names[-1]

# Running 
if(intensity == False):
    res=StainingAnalysis(files=files, marker_names=marker_names, positive=positive, cutoff=cutoff, channels=channels, method=method, output=output)
else:
    res=IntensityAnalysis(files=files, filenames=filenames, marker_names=marker_names, channels=channels)
