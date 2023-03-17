import os, sys
import argparse
import numpy as np
import pandas as pd
import torch
from cellpose import utils, models, io
import matplotlib.pyplot as plt # for plotting
from scipy.ndimage import find_objects, binary_fill_holes # for min_size

if torch.cuda.is_available() :
    gpu = True
else :
    gpu = False

### Set arguments ###
ap = argparse.ArgumentParser()
ap.add_argument('-f', nargs='?', type=str, help='File name')
ap.add_argument('-s', nargs='?', type=str, help='(True/False). Plotting results')
ap.add_argument('-filelist', '--filelist', nargs='*', required=True)
ap.add_argument('-d', nargs='?', type=float, help='Cell diameter')
ap.add_argument('-o', nargs='?', type=float, help='Flow threshold')
ap.add_argument('-m', nargs='?', type=float, help='Min_size')
ap.add_argument('-a', nargs='?', type=float, help='Min_average_intensity cutoff')
ap.add_argument('-t', nargs='?', type=float, help='Min_total_intensity cutoff')
#ap.add_argument('-cellprob', type=float, default=-2, help='Cell probability threshold')
ap.add_argument('-c', nargs='?', type=float, help='Cell probability threshold')
ap.add_argument('-resample', type=bool, default=True, help='(True/False). Resample')
ap.add_argument('-r', nargs='?', type=str, help='(True/False). Cellpose output')
ap.add_argument('-l', nargs='?', type=str, help='Channel')
ap.add_argument('-p', nargs='?', type=str, help='Pre-trained model')
args = ap.parse_args()
if(args.s == 'False'): args.s = False
if(args.s == 'True'): args.s = True
if(args.r == 'False'): args.r = False
if(args.r == 'True'): args.r = True

#print(args.f)
#print(args.filelist)
#print(args)

### Utilities ###
def fill_holes_and_remove_small_masks(masks, min_size=15):
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('fill_holes_and_remove_small_masks takes 2D or 3D array, not %dD array'%masks.ndim)
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i+1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            else:
                if msk.ndim==3:
                    for k in range(msk.shape[0]):
                        msk[k] = binary_fill_holes(msk[k])
                else:
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j+1)
                j+=1
    return masks

class Intensity:
    def __init__(self, image, mask, channels, min_ave_intensity=0, min_total_intensity=0):
        if(channels == [0,0]):
            img = image
        else:
            img = image[:,:,(channels[0]-1)]
        act_idx = np.unique(mask)
        if(sum(act_idx==0) != 0): act_idx = np.delete(act_idx,0) # select masks only (remove 0)
        intensity = []
        for i in act_idx :
            mask_pixel = np.where(mask == i) # mask pixels
            pixel_int = [] # contain pixel intensity
            for j in np.arange(0,len(mask_pixel[0])) :
                pixel_int.append(img[mask_pixel[0][j], mask_pixel[1][j]])
            if(min_ave_intensity != 0): intensity.append(np.mean(pixel_int)) # average intensity for each mask
            if(min_total_intensity != 0): intensity.append(sum(pixel_int)) # total intensity for each mask
        remove_masks_idx = []
        for i in range(len(intensity)):
            if(min_ave_intensity != 0 ):
                if(intensity[i] < min_ave_intensity): remove_masks_idx.append(i+1)
            if(min_total_intensity != 0 ):
                if(intensity[i] < min_total_intensity): remove_masks_idx.append(i+1)
        remove_masks_idx = np.array(remove_masks_idx)
        mask_new = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.int32)
        idx = 1
        for i in act_idx:
            if(sum(i == remove_masks_idx) == 0):
                pixel_new = np.where(mask == i)
                for j in range(len(pixel_new[0])):
                    mask_new[pixel_new[0][j], pixel_new[1][j]] = idx
                idx = idx+1
            else:
                pixel_new = np.where(mask == i)
                for j in range(len(pixel_new[0])):
                    mask_new[pixel_new[0][j], pixel_new[1][j]] = 0
        self.mask = mask_new
        self.intensity = intensity

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



### Running Cellpose ###
def run_cellpose(files, start_path, given_path, gpu, diameter, flow, cellprob, channels, resample, minsize, min_ave_intensity, min_total_intensity, plot, output, pretrained) :
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
        img = io.imread(item); X = img.copy()
        filename = os.path.splitext(item)[0]
        if(pretrained == 'cyto'):
            masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=channels, flow_threshold=flow, cellprob_threshold=cellprob, resample=resample)
        else:
            masks, flows, styles = model.eval(img, diameter=diameter, channels=channels, flow_threshold=flow, cellprob_threshold=cellprob, resample=resample)
            diams = diameter
        
        # Post-processing (min_size, min_intensity) #
        masks = fill_holes_and_remove_small_masks(masks, min_size=minsize) # minsize
        res_intensity = Intensity(image=img, mask=masks, channels=channels, min_ave_intensity=min_ave_intensity, min_total_intensity=min_total_intensity) # intensity
        masks = res_intensity.mask
        
        if(given_path == "None") :
            #ncell.append(masks.max())
            ncell.append((len(np.unique(masks))-1))
        else :
            #print("Cell count: ", masks.max())
            print("Cell count: ", (len(np.unique(masks))-1))
        
        # Save output #
        save_path = start_path + '/' + filename
        if(output == True):
            io.masks_flows_to_seg(img,masks,flows,diams, file_names=save_path + '.npy') # save .npy file
            
            # Size #
            size_masks = []
            act_mask = np.delete(np.unique(masks),0)
            for i in act_mask:
                mask_pixel = np.where(masks == i)
                size_masks.append(len(mask_pixel[0]))
            # XY coordinates #
            outlines = GetCenterCoor(masks)
            mask_res = pd.DataFrame([size_masks,outlines]).T
            mask_res.columns = ["size","xy_coordinate"]
            cellnames = []
            for i in range(mask_res.shape[0]): cellnames.append("Cell_" + str(i+1))
            mask_res.index = cellnames
            mask_res.to_csv(start_path + "/" + filename + "_sizes_coordinates.txt", header=True, index=True, sep=',')
        
        # Save plot #
        if(plot == True) :
            # Mask plot (outliine) #
            my_dpi = 96
            outlines = utils.masks_to_outlines(masks)
            outX, outY = np.nonzero(outlines)
            if(channels == [0,0]):
                imgout = outlines
            else:
                imgout= img.copy()
                imgout[outX, outY] = np.array([255,75,75]) # np.array([255,255,255]) white for severity analysis
            plt.figure(figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi), dpi=my_dpi)
            plt.gca().set_axis_off()
            plt.imshow(imgout)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            if(channels == [0,0]):
                plt.imsave(save_path + "_mask_outline.png", imgout, cmap='gray')
            else:
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
                plt.plot(y_coor[i], x_coor[i], marker='o', color='r', ls='', markersize=2)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(save_path + "_mask_point.png", bbox_inches = 'tight', pad_inches = 0)
            plt.close('all')
    
    if(given_path == "None") :
        ncell_mat = pd.DataFrame(list(zip(files,ncell)))
        ncell_mat.columns = ["File_name","Cell_count"]
        ncell_mat.to_csv(start_path + "/cellpose_counts.txt", header=True, index=None, sep=',')



### Run cellpose ###
start = os.getcwd()

if(args.l == 'None'):
    channels=[3,0]
else:
    channels = args.l
    channels = channels[1:-1] # remove []
    channels = channels.split(",")
    for i in range(len(channels)): channels[i] = int(channels[i])

if(args.p == 'None'):
    pretrained='cytotrain7'
else:
    pretrained = args.p

if(args.f == "None") :
    if(args.filelist[0] == "*.tiff" or args.filelist[0] == "*.png") :
        print("Warning: there is no .tiff or .png file in the path.")
    else : 
        common_path = start
        files = []
        for k in args.filelist :
            total_path = common_path + "/" + k
            file = os.path.relpath(total_path, start)
            files.append(file)
        print(files)
        print(start)
        print(args.f)
        sys.exit("done")
        run_cellpose(files=files, start_path=start, given_path=args.f, gpu=gpu, diameter=args.d, flow=args.o, cellprob=args.c, channels=channels, resample=args.resample, minsize=args.m, min_ave_intensity=args.a, min_total_intensity=args.t, plot=args.s, output=args.r, pretrained=pretrained)
else :
    total_path = args.f
    files = []
    file = os.path.relpath(total_path, start)
    files.append(file)
    run_cellpose(files=files, start_path=start, given_path=args.f, gpu=gpu, diameter=args.d, flow=args.o, cellprob=args.c, channels=channels, resample=args.resample, minsize=args.m, min_ave_intensity=args.a, min_total_intensity=args.t, plot=args.s, output=args.r, pretrained=pretrained)

