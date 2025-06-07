import argparse, glob, os, sys, json
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd

from cellpose import utils, io

from tsp import imread, imsave, image_to_rgb, normalize99
from tsp.masks import roifiles2mask, GetCenterCoor
from tsp.AP import mask2outline, masks_to_outlines, tp_fp_fn, tpfpfn, csi, bias, color_fp_fn, compute_iou, average_dice
from tsp.stitching import dostitch
from tsp.alignment import doalign
from tsp.runcellpose import run_cellpose
from tsp.cellphenotyping import StainingAnalysis
from tsp.intensityanalysis import IntensityAnalysis
from tsp.geom import dist2boundary, region_membership
from tsp.split_dataset import split_dataset_by_class
from scipy import ndimage
import matplotlib.pyplot as plt


import timeit
start_time = timeit.default_timer()


def main():
    
    parser = argparse.ArgumentParser(description='tsp parameters')
    parser.add_argument('action', type=str, help='\
        stitchimages, \
        alignimages, \
        collapseimages, \
        runcellpose, \
        cellphenotyping, \
        dist2boundary, regionmembership, \
        AP, checkprediction, mask2outline, roifiles2mask, overlaymasks, dilatemasks,\
        splitdata')
    
    # for stitchimages
    parser.add_argument('--json', type=str, help='configuration file')
    
    # for alignimages
    parser.add_argument('--ref_image', type=str, help='reference image')

    # for collapse images
    parser.add_argument('--mode', type=str, help='mode of collapsing: max, avg', default='max')

    # for alignment images
    parser.add_argument('--alignmentmode', type=str, help='mode of alignment. See help', default='MOTION_TRANSLATION')

    # for mask-related actions
    parser.add_argument('--mask1', type=str, help='mask file 1')
    parser.add_argument('--mask2', type=str, help='mask file 2')
    parser.add_argument('--maskfile', type=str, help='mask file')
    parser.add_argument('--saveas', type=str, help='save file name for overlaymasks or colortp')
    parser.add_argument('--predfolder', type=str, help='checkprediction prediction folder')
    parser.add_argument('--gtfolder', type=str, help='checkprediction ground truth folder')
    parser.add_argument('--imgfolder', type=str, help='image folder')
    parser.add_argument('--width', type=int, help='width of image', required=False, default=1392)
    parser.add_argument('--height', type=int, help='height of image', required=False, default=1240)
    parser.add_argument('--metric', type=str, help='csi or bias or tpfpfn or coloring', required=False, default='csi')
    parser.add_argument('--dilation', type=int, help='number of pixels to dilate', required=False, default=1240)
    parser.add_argument('--saveoutlineonly', action='store_true', help='save only outlines from rois', required=False)

            
    # for runcellpose prediction
    parser.add_argument('--model', type=str, help='Pre-trained model')
    parser.add_argument('--cellprob', type=float, help='cutoff for cell probability', required=False, default=0) 
    parser.add_argument('--d', type=float, help='Cell diameter', required=False, default=0)
    parser.add_argument('--flow', type=float, help='Flow threshold', required=False, default=0.4)
    parser.add_argument('--normalize99', action='store_true', help='normalize to 1-99 instead of 0-100 percentiles', required=False) # 
    # output control
    parser.add_argument('--saveimgwithmasks', action='store_true', help='save image with masks in mask outline files', required=False) 
    parser.add_argument('--saveflow', action='store_true', help='save flow etc as npy files', required=False) 
    parser.add_argument('--saveroi', action='store_true', help='save masks as roi files', required=False)
    parser.add_argument('--s', action='store_true', help='save additional images with masks plotted as dots and fills and numbered outlines', required=False) 
    
    parser.add_argument('--color', type=str, help='checkprediction image folder')

    # for cellphenotyping 
    parser.add_argument('--m', type=str, help='(Mask/Intensity_mean/Intensity_total/Intensity_pos/Intensity_median)')
    parser.add_argument('--c', type=str, help='cutoff') 
    parser.add_argument('--c2', type=str, help='cutoff 2', required=False) 
    parser.add_argument('--pixel_pos_threshold', type=str, help='', required=False) 
    parser.add_argument('--p', type=str, help='(True/False). Positive or Negative')
    parser.add_argument('--n', type=str, help='marker names')
    parser.add_argument('--mask_dilations', type=str, help='for intensity, we can specify the numbers of pixels to dilate on all sides. A negative number means shrinking', required=False)
            
    # shared
    parser.add_argument('--f', type=str, help='files') 
    parser.add_argument('--l', type=str, help='Channel', required=False)
    parser.add_argument('--min_size', type=int, help='minimal size of masks', required=False, default=15)
    parser.add_argument('--min_totalintensity', type=int, help='minimal value of total intensity', required=False, default=0)
    parser.add_argument('--min_avgintensity', type=int, help='minimal value of average intensity', required=False, default=0)    
    parser.add_argument('--imagefile', type=str, help='image file')
    parser.add_argument('--verbose', action='store_true', help='show information about running and settings and save to log', required=False)    

    # dist2boundary
    parser.add_argument('--cells', type=str, help='csv files containing the cell center coordinates', required=False)
    parser.add_argument('--boundaryroi', type=str, help='roi files containing the boundary lines', required=False)
    
    # regionmembership
    parser.add_argument('--regionroi', type=str, help='roi files defining regions, e.g. [region1.roi,region2.roi]', required=False)
    
    # splitdata
    parser.add_argument('--dataset', type=str, help='dataset folder', required=False)
    parser.add_argument('--imgext', type=str, help='image file extension, e.g. jpg, png, tif', required=False)
    parser.add_argument('--trainratio', type=str, help='train to all, e.g. 0.9', required=False)
    
    args = parser.parse_args()

    
    if args.action=='runcellpose':
        if args.f == None or args.model == None or args.l == None :
            sys.exit("ERROR: --f, --model, --l are required arguments")            
        
        files = args.f[1:-1].split(",") if args.f[0]=='[' else glob.glob(args.f)
        if len(files)==0: 
            sys.exit ("no files found")
        else:
            print('working on: ', end=" ")
            print(files)
        
        channels = [int(i) for i in args.l[1:-1].split(",")]

        # pretrained="cytotrain7"; diameter=0.; flow=0.4; cellprob=0.; min_size=10; min_avg_intensity=10; min_total_intensity=10; plot=False; output=False; channels=[3,0]
        run_cellpose(files=files, 
                     channels=channels,
                     pretrained=args.model, 
                     diameter=args.d, flow=args.flow, cellprob=args.cellprob, 
                     normalize_100=not args.normalize99,
                     min_size=args.min_size, min_avg_intensity=args.min_avgintensity, min_total_intensity=args.min_totalintensity, 
                     save_outlines_only=not args.saveimgwithmasks, save_additional_plots=args.s, save_roi=args.saveroi, save_flow=args.saveflow) 
        

        
    elif args.action=='cellphenotyping':        
        #files=["M872956_JML_Position8_CD3_img.png","M872956_JML_Position8_CD4_img.png","M872956_JML_Position8_CD8_img.png"]; marker_names=["CD4","CD8"]; positives=[True,False]; cutoffs=[.5,.5]; channels=None; methods=["Mask","Mask"]; save_plot=True
        #files=["JM_Les_CD3_stitched_gray_alignedtoCD45_m_cytotrain7.png","JM_Les_CD8a_stitched_gray_alignedtoCD45_m_cytotrain7.png"]; marker_names=["CD8"]; positives=[True]; cutoffs=[.5]; cutoffs2=[.9]; channels=None; methods=["Mask","Mask"]; save_plot=True
                
        # get a list of file names from --f
        files = args.f[1:-1].split(",") 
        if len(files)==0: sys.exit ("no files found")
        else: 
            print('working on: ', end=" ")
            print(files)
            
        positives = args.p[1:-1].split(",") 
        cutoffs = [float(c) for c in args.c[1:-1].split(",")]
        methods = args.m[1:-1].split(",")             
        marker_names = args.n[1:-1].split(",")
        channels = [int(i)-1 for i in args.l[1:-1].split(",")] if args.l is not None else None
        
        if args.c2 is not None:
            cutoffs2 = [float(c) for c in args.c2[1:-1].split(",") ]
        else:
            cutoffs2 = [1 for c in cutoffs]
        
        if args.mask_dilations is not None:
            mask_dilations = [int(c) for c in args.mask_dilations[1:-1].split(",") ]
        else:
            mask_dilations = [0 for c in cutoffs]
        
        if args.pixel_pos_threshold is not None:
            pixel_pos_thresholds = [float(c) for c in args.pixel_pos_threshold[1:-1].split(",") ]
        else:
            pixel_pos_thresholds = [100 for c in cutoffs]
        
        StainingAnalysis(files=files, marker_names=marker_names, positives=[p=='True' for p in positives], cutoffs=cutoffs, channels=channels, methods=methods, save_plot=args.s, cutoffs2=cutoffs2, pixel_pos_thresholds=pixel_pos_thresholds, mask_dilations=mask_dilations)
        
        
    elif args.action=='intensityanalysis':        
        if args.f == None or args.maskfile == None:
            sys.exit("ERROR: --f and --maskfile are required arguments")            

        # get a list of file names from --f
        files = args.f[1:-1].split(",") if args.f[0]=='[' else glob.glob(args.f)
        if len(files)==0: sys.exit ("no files found")
        else: 
            print('working on: ', end=" ")
            print(files)
                    
        channel = int(args.l)-1 if args.l is not None else None
        
        IntensityAnalysis(mask_file=args.maskfile, image_files=files, channel=channel)
        
        
    elif args.action=='regionmembership':
        if args.f == None or args.regionroi == None:
            sys.exit("ERROR: --f, --regionroi are required arguments")            
        
        # get a list of file names from --f
        files = args.f[1:-1].split(",") if args.f[0]=='[' else glob.glob(args.f)
        if len(files)==0: sys.exit ("no files found")
        else: 
            print('working on: ', end=" ")
            print(files)
                    
        region_roi_files = args.regionroi[1:-1].split(",") if args.regionroi[0]=='[' else glob.glob(args.regionroi)
        print(region_roi_files)

        region_membership(files, region_roi_files)

        
    elif args.action=='dist2boundary':
        if args.f == None or args.boundaryroi == None:
            sys.exit("ERROR: --f, --boundaryroi are required arguments")            

        # get a list of file names from --f
        files = args.f[1:-1].split(",") if args.f[0]=='[' else glob.glob(args.f)
        if len(files)==0: sys.exit ("no files found")
        else: 
            print('working on: ', end=" ")
            print(files)
                    
        roi_files = args.boundaryroi[1:-1].split(",") if args.boundaryroi[0]=='[' else glob.glob(args.boundaryroi)
        if len(roi_files)==0: sys.exit ("no roi_files found")
        else: 
            print('working on: ', end=" ")
            print(roi_files)
        
        dist2boundary(files, roi_files)

        
    elif args.action=="stitchimages":
        if args.json == None or args.imgfolder==None:
            sys.exit("ERROR: --json required")            

        # with open('D:/DeepLearning/images_from_K/cell_phenotyping/Cellscape_layout_by_percentage.json') as config_file:
        with open(args.json) as config_file:
            config = json.load(config_file)
        
        dostitch(config, args.imgfolder+"/")


    elif args.action=="alignimages":
        channels = [int(i)-1 for i in args.l[1:-1].split(",")] if args.l is not None else None
        
        # get a list of file names from --f
        files = args.f[1:-1].split(",") if args.f[0]=='[' else glob.glob(args.f)
        if len(files)==0: sys.exit ("no files found")
        else: 
            print('working on: ', end=" ")
            print(files)
        
        # remove ref_image from files
        if args.ref_image in files:
            files.remove(args.ref_image)
                    
        for f in files:
            doalign (args.ref_image, f, channels, args.alignmentmode)


    elif args.action=="collapseimages":
        if args.f == None or args.saveas == None:
            sys.exit("ERROR: --f and --saveas are required")            

        # get a list of file names from --f
        files = args.f[1:-1].split(",") if args.f[0]=='[' else glob.glob(args.f)
        if len(files)==0: sys.exit ("no files found")
        else: 
            print('working on: ', end=" ")
            print(files)
                    
        im0 = imread(files[0])
        im = im0
        
        for i in range(1,len(files)):
            im_next = imread(files[i])
            if args.mode == 'max':
                im = np.maximum(im, im_next)
            elif args.mode == 'avg':
                # recompute average
                im = ((im0*i + im_next)/(i+1)).astype(np.uint8)
            else:
                sys.exit("ERROR: --mode needs to be max or avg")            

        imsave(args.saveas,  im)


    elif args.action=='mask2outline':
        if args.f == None:
            sys.exit("ERROR: --f is required arguments")            
        
        # get a list of file names from --f
        files = args.f[1:-1].split(",") if args.f[0]=='[' else glob.glob(args.f)
        if len(files)==0: sys.exit ("no files found")
        else: 
            print('working on: ', end=" ")
            print(files)
        
        mask_color = args.color
        if mask_color is not None:
            mask_color=mask_color[1:-1].split(",")
                    
        for mask_file in files:
            mask2outline(mask_file, mask_color)
                
                
    elif args.action=="roifiles2mask":
        if args.f == None or args.saveas == None:
            sys.exit("ERROR: --f, --saveas are required arguments")            

        # get a list of file names from --f
        files = args.f[1:-1].split(",") if args.f[0]=='[' else glob.glob(args.f)
        if len(files)==0: sys.exit ("no files found")
        else: 
            print('working on: ', end=" ")
            print(files)
                    
        if args.imagefile is not None:
            img = imread(args.imagefile)
            height, width = img.shape[0:2]
        else:
            height, width = args.height, args.width
        
        roifiles2mask (files, width, height, saveas=args.saveas, outline=True, fill=not args.saveoutlineonly)


    elif args.action=='dilatemasks':
        if args.dilation == None:
            sys.exit("ERROR: --dilation is a required argument")            
        dilation=args.dilation    
        
        masks  =imread(args.maskfile)    
        mask_indices = np.unique(masks)[1:]
                
        if dilation!=0:
            # Generate a structure element (kernel) for erosion
            # This creates a 2D kernel for 2D images; adjust dimensions for 3D images if necessary
            structure_element = np.ones((2*abs(dilation)+1, 2*abs(dilation)+1))
    
            mod_masks = np.zeros_like(masks)
            
            for mask_value in mask_indices:
                # Create a binary mask for the current value
                print(mask_value)
                binary_mask = (masks == mask_value)
                
                # Apply binary dilation/erosion
                if dilation>0:
                    mod_binary_mask = ndimage.binary_dilation(binary_mask, structure=structure_element).astype(binary_mask.dtype)
                else:                      
                    mod_binary_mask = ndimage.binary_erosion(binary_mask, structure=structure_element).astype(binary_mask.dtype)
                
                # Combine mod masks, assuming non-overlapping masks for simplicity
                mod_masks += mod_binary_mask * mask_value
        else:
            mod_masks = masks
            
        filename=os.path.splitext(args.maskfile)[0]
        fileext=os.path.splitext(args.maskfile)[1]
        imsave(filename+"_d"+str(dilation)+fileext, mod_masks)    

        # save an outline file
        outlines = masks_to_outlines(mod_masks)
        plt.imsave(filename+"_d"+str(dilation)+"_o"+fileext, outlines, cmap='gray')
        
        # save _cp_outline to convert to roi by ImageJ
        outlines_list = utils.outlines_list(mod_masks)
        io.outlines_to_text(filename, outlines_list)


        # ## Save a csv file of mask info. One row per mask, columns include size, center_x, center_y
        # centers=GetCenterCoor(mod_masks)
        # y_coor, x_coor = zip(*centers)
        #         # turn tuples into arrays to use as.type later
        # y_coor=np.array(y_coor); x_coor=np.array(x_coor)
        # # get size
        # tmp=np.unique(mod_masks, return_counts=True)
        # sizes = tmp[1][1:]#.tolist()    # keep it as an array     
        # ncell=len(sizes)
        # # make a data frame
        # mask_info = pd.DataFrame({
        #     "center_x": x_coor, 
        #     "center_y": y_coor,
        #     "size": sizes
        # })
        # mask_info.index = [f"Cell_{i}" for i in range(1,ncell+1)]
        # mask_info=mask_info.round().astype(int)
        # mask_info.to_csv(filename + "_d"+str(dilation) +".csv", header=True, index=True, sep=',')

                            

    elif args.action=='overlaymasks':
        # add masks to images    
        img  =imread(args.imagefile)
        imgout = image_to_rgb(normalize99(img))
        
        if args.mask1:
            masks=imread(args.mask1)    
            outlines = masks_to_outlines(masks)
            outX, outY = np.nonzero(outlines)
            imgout[outX, outY] = np.array([255,0,0]) # pure red

        if args.mask2:
            masks=imread(args.mask2)    
            outlines = masks_to_outlines(masks)
            outX, outY = np.nonzero(outlines)
            imgout[outX, outY] = np.array([0,255,0]) # pure green
        
        # one more time and turn overlap into yellow
        if args.mask1:
            masks=imread(args.mask1)    
            outlines = masks_to_outlines(masks)
            outX, outY = np.nonzero(outlines)
            imgout[outX, outY, 0] = 255
        
        if args.saveas:
            newfilename=args.imagefile.replace("_img.png","_img_{}.png".format(args.saveas))
        else:
            newfilename=args.imagefile.replace("_img.png","_img_masksadded.png")
        imsave(newfilename,  imgout)


    elif args.action=='AP':
        filename1, file_extension1 = os.path.splitext(args.mask1)
        if file_extension1==".png":
            mask1=imread(args.mask1)
        elif file_extension1==".npz":
            mask1 = np.load(args.mask1, allow_pickle=True)
            mask1 = mask1['masks']
        else:
            print("file type not supported: "+file_extension1)
            
        filename2, file_extension2 = os.path.splitext(args.mask2)
        if file_extension2==".png":
            mask2=imread(args.mask2)
        elif file_extension2==".npz":
            mask2 = np.load(args.mask2, allow_pickle=True)
            mask2 = mask2['masks']
        
        out=csi(mask1, mask2)
        print('{:.3}'.format(out))
        
        
    elif args.action=='checkprediction':
        gt_file_names = sorted(os.listdir(args.gtfolder)) # file names only, no path        
        thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
        res_mat = []
        csi_vec=[]
        for gt_file_name in gt_file_names:
            img_name = gt_file_name.split('_masks')[0]
            if args.verbose: print(img_name, end="\t")
            gt_path = sorted(glob.glob(args.gtfolder+'/'+img_name+"*"))[0] 
            pred_path = sorted(glob.glob(args.predfolder+'/'+img_name+"*"))[0] 
            y_pred = imread(pred_path)
            labels = imread(gt_path)
            if args.imgfolder:
                img_path = sorted(glob.glob(args.imgfolder+'/'+img_name+"*"))[0] 
                img  =imread(img_path)
                imgout = image_to_rgb(normalize99(img))
            
    
            true_objects = np.unique(labels)
            pred_objects = np.unique(y_pred)
            if args.verbose: print(f"# gt: {len(true_objects)},", end=" ")

            # filter masks based on minimal size of masks
            if args.min_size>0:
                area_true = np.histogram(labels, bins=np.append(true_objects, np.inf))[0]
                area_pred = np.histogram(y_pred, bins=np.append(pred_objects, np.inf))[0]
                true_objects1 = true_objects[area_true>=args.min_size]
                if args.verbose: print(f"# gt: {len(true_objects1)},", end=" ")
                pred_objects1 = pred_objects[area_pred>=args.min_size]
                for idx in true_objects:
                    if not (idx in true_objects1):
                        temp = np.where(labels == idx)
                        labels[temp[0], temp[1]] = 0
                for idx in pred_objects:
                    if not (idx in pred_objects1):
                        temp = np.where(y_pred == idx)
                        y_pred[temp[0], temp[1]] = 0
        
            # filter masks based on minimal total intensity
            if args.min_totalintensity>0:                
                totalintensity_true=[]
                for idx in true_objects:
                    temp = np.where(labels == idx)
                    totalintensity_true.append(sum(img[temp[0], temp[1]]))
                totalintensity_pred=[]
                for idx in pred_objects:
                    temp = np.where(y_pred == idx)
                    totalintensity_pred.append(sum(img[temp[0], temp[1]]))

                true_objects1 = true_objects[np.array(totalintensity_true)>=args.min_totalintensity]
                if args.verbose: print(f"# gt: {len(true_objects1)},", end=" ")
                pred_objects1 = pred_objects[np.array(totalintensity_pred)>=args.min_totalintensity]
                for idx in true_objects:
                    if not (idx in true_objects1):
                        temp = np.where(labels == idx)
                        labels[temp[0], temp[1]] = 0
                for idx in pred_objects:
                    if not (idx in pred_objects1):
                        temp = np.where(y_pred == idx)
                        y_pred[temp[0], temp[1]] = 0
        
            # filter masks based on minimal average intensity
            if args.min_avgintensity>0:
                area_true = np.histogram(labels, bins=np.append(true_objects, np.inf))[0]
                area_pred = np.histogram(y_pred, bins=np.append(pred_objects, np.inf))[0]
                
                avgintensity_true=[]
                for i, idx in enumerate(true_objects):
                    temp = np.where(labels == idx)
                    avgintensity_true.append(sum(img[temp[0], temp[1]])/area_true[i])
                avgintensity_pred=[]
                for i, idx in enumerate(pred_objects):
                    temp = np.where(y_pred == idx)
                    avgintensity_pred.append(sum(img[temp[0], temp[1]])/area_pred[i])

                true_objects1 = true_objects[np.array(avgintensity_true)>=args.min_avgintensity]
                if args.verbose: print(f"# gt: {len(true_objects1)},", end=" ")
                pred_objects1 = pred_objects[np.array(avgintensity_pred)>=args.min_avgintensity]
                for idx in true_objects:
                    if not (idx in true_objects1):
                        temp = np.where(labels == idx)
                        labels[temp[0], temp[1]] = 0
                for idx in pred_objects:
                    if not (idx in pred_objects1):
                        temp = np.where(y_pred == idx)
                        y_pred[temp[0], temp[1]] = 0
        
            
            tpfpfn_vec = tpfpfn(labels, y_pred, threshold=0.5) 
            csi_5 = csi(labels, y_pred, threshold=0.5)
            csi_vec.append(csi_5)
            if args.verbose: print("csi " + "{0:0.3f}".format(csi_5) + " tp,fp,fn:", ' '.join(["{0:0.0f}".format(i) for i in tpfpfn_vec]))
            if args.verbose: print(f"mAP={np.mean(csi_vec)}")        

            if args.metric=='bias':
                res_temp = bias(labels, y_pred)
                res_mat.append(round(res_temp,5))
            elif args.metric=='ari':
                res_temp = adjusted_rand_score(labels.flatten(), y_pred.flatten())
                res_mat.append(round(res_temp,5))
            elif args.metric=='dice':
                res_temp = average_dice(labels, y_pred)
                res_mat.append(round(res_temp,5))
            elif args.metric=='csi': 
                res_vec = []
                for t in thresholds:
                    res_temp = csi(labels, y_pred, threshold=t) 
                    res_vec.append(round(res_temp,6))
                res_mat.append(res_vec)
            elif args.metric=='tpfpfn': 
                res_mat.append(tpfpfn_vec)
            elif args.metric=='coloring':
                color_fp_fn(gt_path, pred_path)
            elif args.metric=='colortp':
                # add masks to images, color tp yellow and fp green                
                # labels_idx = np.setdiff1d(np.unique(labels), np.array([0])) # remove background 0
                y_pred_idx = np.setdiff1d(np.unique(y_pred), np.array([0])) # remove background 0
            
                # paint all y_predicted labelss yellow
                outlines = masks_to_outlines(y_pred)
                outX, outY = np.nonzero(outlines)
                imgout[outX, outY] = np.array([255,255,0]) 
                
                # plot non-fp y_predicted labelss green
                iou = compute_iou(mask_true=labels, mask_pred=y_pred) # compute iou
                tp, fp, fn = tp_fp_fn(threshold=0.5, iou=iou, index=True)
                fp_idx = y_pred_idx[fp]
        
                y_pred_nfp = y_pred.copy()
                for idx in y_pred_idx:
                    if not (idx in fp_idx):
                        temp = np.where(y_pred_nfp == idx)
                        y_pred_nfp[temp[0], temp[1]] = 0
                nfp_outlines = masks_to_outlines(y_pred_nfp)
                
                res = imgout
                outX, outY = np.nonzero(nfp_outlines)
                res[outX, outY, 0] = 0
                
                # save files
                if args.saveas:
                    newfilename=img_name+"_img_{}.png".format(args.saveas)
                else:
                    newfilename=img_name+"_img_labelssadded.png"
                imsave(newfilename,  res)
        
                
        if args.metric in ['bias', 'ari', 'dice']:
            res_temp = np.array([res_mat])
            print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format
        elif args.metric=='ari':
            res_temp = np.array([res_mat])
            print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format
        elif args.metric=='csi':
            print("APs at threshold of 0.5-1.0")
            print(res_mat)
            res_temp = list(list(zip(*res_mat))[0]) # AP at threshold of 0.5
            # res_temp.append(np.mean(res_temp))
            res_temp = np.array([res_temp]) 
            #print(" \\\\\n".join([" & ".join(map(str,line)) for line in res_temp])) # latex table format
            print("APs at threshold of 0.5")
            print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format
        elif args.metric=='tpfpfn':
            res_temp = np.array([res_mat])
            print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format
        
    elif args.action=='splitdata':
        imgext=args.imgext if args.imgext is not None else 'tif'
        trainratio=float(args.trainratio if args.trainratio is not None else '0.9')
        split_dataset_by_class(args.dataset, imgext, trainratio)
        
    else:
            print("Wrong action \n")
        
    if args.verbose: print(f"time passed: {(timeit.default_timer() - start_time)/60:.1f} min"); 

if __name__ == '__main__':
    main()
    
