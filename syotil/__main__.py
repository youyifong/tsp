import argparse, glob
import numpy as np
from syotil import *
# this function differs from cellpose.imread, which does additional things like 
# if img.ndim > 2: img[..., [2,1,0]], which reverses the order of the last dimension, which is the color channel
from skimage.io import imread 
# cv2.imread handle will make the masks 3 channel
import os


def main():
    
    parser = argparse.ArgumentParser(description='syotil parameters')
    parser.add_argument('action', type=str, help='AP, maskfile2outline, checkprediction, overlaymasks, roifiles2mask')
    parser.add_argument('--mask1', 
                        type=str, help='mask file 1 for AP or overlaymasks', required=False)
    parser.add_argument('--mask2', 
                        type=str, help='mask file 2 for AP or overlaymasks', required=False)
    parser.add_argument('--maskfile', 
                        type=str, help='mask file for maskfile2outline', required=False)
    parser.add_argument('--imagefile', 
                        type=str, help='image file for overlaymasks', required=False)
    parser.add_argument('--saveas', 
                        type=str, help='save file name for overlaymasks', required=False)
    parser.add_argument('--predfolder', 
                        type=str, help='checkprediction prediction folder', required=False)
    parser.add_argument('--gtfolder', 
                        type=str, help='checkprediction ground truth folder', required=False)
    parser.add_argument('--roifolder', 
                        type=str, help='folder that contains the roi files for roifiles2mask, e.g. M926910_Pos6_RoiSet_49', required=False)
    parser.add_argument('--width', 
                        type=int, help='width of image', required=False, default=1392)
    parser.add_argument('--height', 
                        type=int, help='height of image', required=False, default=1240)
    parser.add_argument('--metric', 
                        default='csi', type=str, help='csi or bias or tpfpfn or coloring', required=False)    
    parser.add_argument('--verbose', action='store_true', help='show information about running and settings and save to log')    
    args = parser.parse_args()


    if args.action=='maskfile2outline':
        filename, extension = os.path.splitext(args.maskfile)
        if extension:
            maskfile2outline(args.maskfile)
        else:
            for i in os.listdir():
                maskfile2outline(i)
                
    elif args.action=="roifiles2mask":
        roifiles2mask(args.roifolder+"/*", args.width, args.height)

    elif args.action=='overlaymasks':
        # add masks to images    
        img  =imread(args.imagefile)
        img0 = normalize99(img)
        imgout = image_to_rgb(img0)
        
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
        for gt_file_name in gt_file_names:
            img_name = gt_file_name.split('_masks')[0]
            gt_path = sorted(glob.glob(args.gtfolder+'/'+img_name+"*"))[0] 
            pred_path = sorted(glob.glob(args.predfolder+'/'+img_name+"*"))[0] 
                        
            y_pred = imread(pred_path)
            labels = imread(gt_path)
            
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
                res_vec = tpfpfn(labels, y_pred, threshold=0.5) 
                res_mat.append(res_vec)
            elif args.metric=='coloring':
                color_fp_fn(gt_path, pred_path)
                        
        if args.metric=='bias':
            res_temp = np.array([res_mat])
            print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format
        elif args.metric=='csi':
            #APs at threshold of 0.5
            res_temp = list(list(zip(*res_mat))[0]) # AP at threshold of 0.5
            res_temp = np.array([res_temp]) 
            #print(" \\\\\n".join([" & ".join(map(str,line)) for line in res_temp])) # latex table format
            print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format
        elif args.metric=='tpfpfn':
            res_temp = np.array([res_mat])
            print (', '.join(pred_name))
            print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format
        


if __name__ == '__main__':
    main()
    
