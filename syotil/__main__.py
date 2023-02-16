"""
roifiles2mask
    roifolder: path to the folder containing the roi files
    height and width: dimension of the image, default to 1240x1392
the function creates two files, one mask png file and one mask outline png file.


"""


import argparse, glob
import numpy as np
from syotil import *
# this function differs from cellpose.imread, which does additional things like 
# if img.ndim > 2: img[..., [2,1,0]], which reverses the order of the last dimension, which is the color channel
from skimage.io import imread 
# cv2.imread handle will make the masks 3 channel
import os

# for alignment
import pyelastix

def main():
    
    parser = argparse.ArgumentParser(description='syotil parameters')
    parser.add_argument('action', type=str, help='AP, maskfile2outline, checkprediction, overlaymasks, roifiles2mask, alignimages')
    # overlaymasks
        # add mask1 in red, mask2 in green (optional), and overlap in yellow, all on top of images
    # colortp
        # add mask2 in green and highlight tp (based on comparing with mask1) in yellow, on top of images
    # roifiles2mask --roifolder   --width   --height  
        # makes masks png file
    # maskfile2outline --maskfile 
        # makes outlines
    # checkprediction --metric   --predfolder   --gtfolder   --min_size
    
    # alignimages --ref_image xx  --image2 xx
        # align image2 to ref_image with elastic alignment. requires elastix executable
    
        
    parser.add_argument('--mask1', 
                        type=str, help='mask file 1 for AP or overlaymasks', required=False)
    parser.add_argument('--mask2', 
                        type=str, help='mask file 2 for AP or overlaymasks', required=False)
    parser.add_argument('--ref_image', 
                        type=str, help='reference image', required=False)
    parser.add_argument('--image2', 
                        type=str, help='image file 2', required=False)
    parser.add_argument('--maskfile', 
                        type=str, help='mask file for maskfile2outline', required=False)
    parser.add_argument('--imagefile', 
                        type=str, help='image file for overlaymasks', required=False)
    parser.add_argument('--saveas', 
                        type=str, help='save file name for overlaymasks or colortp', required=False)
    parser.add_argument('--predfolder', 
                        type=str, help='checkprediction prediction folder', required=False)
    parser.add_argument('--gtfolder', 
                        type=str, help='checkprediction ground truth folder', required=False)
    parser.add_argument('--imgfolder', 
                        type=str, help='checkprediction image folder', required=False)
    parser.add_argument('--roifolder', 
                        type=str, help='folder that contains the roi files for roifiles2mask, e.g. M926910_Pos6_RoiSet_49', required=False)
    parser.add_argument('--width', 
                        type=int, help='width of image', required=False, default=1392)
    parser.add_argument('--height', 
                        type=int, help='height of image', required=False, default=1240)
    parser.add_argument('--min_size', 
                        type=int, help='minimal size of masks', required=False, default=0)
    parser.add_argument('--min_totalintensity', 
                        type=int, help='minimal value of total intensity', required=False, default=0)
    parser.add_argument('--min_avgintensity', 
                        type=int, help='minimal value of average intensity', required=False, default=0)
    parser.add_argument('--attachment', 
                        type=int, help='alignment parameter', required=False, default=5)
    parser.add_argument('--tightness', 
                        type=float, help='alignment parameter', required=False, default=0.7)
    parser.add_argument('--metric', 
                        default='csi', type=str, help='csi or bias or tpfpfn or coloring', required=False)    
    parser.add_argument('--verbose', action='store_true', help='show information about running and settings and save to log')    
    args = parser.parse_args()


    if args.action=='maskfile2outline':
        filename, extension = os.path.splitext(args.maskfile)
        if extension:
            maskfile2outline(args.maskfile)
        else:
            for i in os.listdir(args.maskfile):
                maskfile2outline(args.maskfile+"/"+i)
                
    elif args.action=="roifiles2mask":
        roifiles2mask(args.roifolder+"/*", args.width, args.height)

    elif args.action=="alignimages":
        filename, file_extension = os.path.splitext(args.image2)
        image1=imread(args.ref_image)
        image2=imread(args.image2)
        image2_max = np.max(image2)

        # Convert the images to gray level: color is not supported.
        image1 = rgb2gray(image1)
        image2 = rgb2gray(image2)

        # Get params and change a few values
        params = pyelastix.get_default_params(type="AFFINE")
        # params.MaximumNumberOfIterations = 200
        # params.FinalGridSpacingInVoxels = 10
        
        # Apply the registration (im1 and im2 can be 2D or 3D)
        image2_warp, field = pyelastix.register(image2, image1, params)
                
        # save to file
        image2_warp=image_to_rgb(image2_warp)
        imsave(filename+"_aligned"+file_extension,  image2_warp)

        
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
            print(img_name, end="\t")
            gt_path = sorted(glob.glob(args.gtfolder+'/'+img_name+"*"))[0] 
            # print(args.predfolder+'/'+img_name+"*")
            pred_path = sorted(glob.glob(args.predfolder+'/'+img_name+"*"))[0] 
            y_pred = imread(pred_path)
            labels = imread(gt_path)
            if args.imgfolder:
                img_path = sorted(glob.glob(args.imgfolder+'/'+img_name+"*"))[0] 
                img  =imread(img_path)
                imgout = image_to_rgb(normalize99(img))
            
    
            true_objects = np.unique(labels)
            pred_objects = np.unique(y_pred)
            print(f"# gt: {len(true_objects)},", end=" ")

            # filter masks based on minimal size of masks
            if args.min_size>0:
                area_true = np.histogram(labels, bins=np.append(true_objects, np.inf))[0]
                area_pred = np.histogram(y_pred, bins=np.append(pred_objects, np.inf))[0]
                true_objects1 = true_objects[area_true>=args.min_size]
                print(f"# gt: {len(true_objects1)},", end=" ")
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
                print(f"# gt: {len(true_objects1)},", end=" ")
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
                print(f"# gt: {len(true_objects1)},", end=" ")
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
            print("csi " + "{0:0.3f}".format(csi_5) + " tp,fp,fn:", ' '.join(["{0:0.0f}".format(i) for i in tpfpfn_vec]))
            
            if args.metric=='bias':
                res_temp = bias(labels, y_pred)
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
                labels_idx = np.setdiff1d(np.unique(labels), np.array([0])) # remove background 0
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
        
        print(f"mAP={np.mean(csi_vec)}")        
                
        if args.metric=='bias':
            res_temp = np.array([res_mat])
            print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format
        elif args.metric=='csi':
            #APs at threshold of 0.5
            res_temp = list(list(zip(*res_mat))[0]) # AP at threshold of 0.5
            # res_temp.append(np.mean(res_temp))
            res_temp = np.array([res_temp]) 
            #print(" \\\\\n".join([" & ".join(map(str,line)) for line in res_temp])) # latex table format
            print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format
        elif args.metric=='tpfpfn':
            res_temp = np.array([res_mat])
            print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format
        


if __name__ == '__main__':
    main()
    
