import argparse, glob, os, sys
import numpy as np
from tsp import imread, imsave, image_to_rgb, normalize99
from tsp.masks import maskfile2outline, roifiles2mask, masks_to_outlines, tp_fp_fn, tpfpfn, csi, bias, color_fp_fn, compute_iou
from tsp.alignment import doalign
from tsp.runcellpose import run_cellpose
from tsp.cellphenotyping import StainingAnalysis
from tsp.intensityanalysis import IntensityAnalysis
from tsp.geom import dist2boundary, region_membership

def main():
    
    parser = argparse.ArgumentParser(description='tsp parameters')
    parser.add_argument('action', type=str, help='\
        alignimages, \
        runcellpose, \
        cellphenotyping, \
        dist2boundary, regionmembership, \
        AP, checkprediction, maskfile2outline, roifiles2mask, overlaymasks')
    
    # for alignimages
    parser.add_argument('--ref_image', type=str, help='reference image')
    parser.add_argument('--image2', type=str, help='image file 2')
    
    # for mask-related actions
    parser.add_argument('--mask1', type=str, help='mask file 1')
    parser.add_argument('--mask2', type=str, help='mask file 2')
    parser.add_argument('--maskfile', type=str, help='mask file for maskfile2outline')
    parser.add_argument('--saveas', type=str, help='save file name for overlaymasks or colortp')
    parser.add_argument('--predfolder', type=str, help='checkprediction prediction folder')
    parser.add_argument('--gtfolder', type=str, help='checkprediction ground truth folder')
    parser.add_argument('--imgfolder', type=str, help='checkprediction image folder')
    parser.add_argument('--roifolder', type=str, help='folder that contains the roi files for roifiles2mask, e.g. M926910_Pos6_RoiSet_49')
    parser.add_argument('--width', type=int, help='width of image', required=False, default=1392)
    parser.add_argument('--height', type=int, help='height of image', required=False, default=1240)
    parser.add_argument('--metric', type=str, help='csi or bias or tpfpfn or coloring', required=False, default='csi')
    
    # shared by runcellpose and cellphenotyping, and intensityanalysis
    parser.add_argument('--f', type=str, help='pattern (cellphenotyping) or file names (runcellpose, intensityanalysis)') 
    parser.add_argument('--r', action='store_true', help='save mask info (runcellpose) or double-stained-mask (cellphenotyping)', required=False) 
        
    # for runcellpose 
    parser.add_argument('--model', type=str, help='Pre-trained model')
    parser.add_argument('--cellprob', type=float, help='cutoff for cell probability', required=False, default=0) 
    parser.add_argument('--d', type=float, help='Cell diameter', required=False, default=0)
    parser.add_argument('--o', type=float, help='Flow threshold', required=False, default=0.4)
    parser.add_argument('--s', action='store_true', help='plot results', required=False) # saves 4 types of mask png files: outline, text, point, fill
    
    # for cellphenotyping 
    parser.add_argument('--m', type=str, help='(Mask/Intensity_avg/Intensity_total)')
    parser.add_argument('--c', type=str, help='cutoff') 
    parser.add_argument('--p', type=str, help='(True/False). Positive or Negative')
    parser.add_argument('--n', type=str, help='marker names')
            
    # shared
    parser.add_argument('--l', type=str, help='Channel', required=False)
    parser.add_argument('--min_size', type=int, help='minimal size of masks', required=False, default=0)
    parser.add_argument('--min_totalintensity', type=int, help='minimal value of total intensity', required=False, default=0)
    parser.add_argument('--min_avgintensity', type=int, help='minimal value of average intensity', required=False, default=0)    
    parser.add_argument('--imagefile', type=str, help='image file')
    parser.add_argument('--verbose', action='store_true', help='show information about running and settings and save to log', required=False)    

    # dist2boundary
    parser.add_argument('--cells', type=str, help='a csv file containing the cell center coordinates', required=False)
    parser.add_argument('--boundaryroi', type=str, help='roi file containing the line', required=False)
    
    # regionmembership
    parser.add_argument('--regionroi', type=str, help='roi files defining regions, e.g. [region1.roi,region2.roi]', required=False)
    
    args = parser.parse_args()

    
    if args.action=='regionmembership':
        if args.cells == None or args.regionroi == None:
            sys.exit("ERROR: --cells, --regionroi are required arguments")            
        
        region_roi_files = args.regionroi[1:-1].split(",")         

        region_membership(args.cells, region_roi_files)

        
    elif args.action=='dist2boundary':
        if args.cells == None or args.boundaryroi == None:
            sys.exit("ERROR: --cells, --boundaryroi are required arguments")            
        dist2boundary(args.cells, args.boundaryroi)

        
    elif args.action=='runcellpose':
        if args.f == None or args.model == None or args.l == None :
            sys.exit("ERROR: --f, --model, --l are required arguments")            
        
        files = glob.glob(args.f)
        channels = [int(i) for i in args.l[1:-1].split(",")]
        print('working on: ', end=" ")
        print(files)
        
        # pretrained="cytotrain7"; diameter=0.; flow=0.4; cellprob=0.; minsize=0; min_ave_intensity=0; min_total_intensity=0; plot=False; output=False
        run_cellpose(files=files, 
                     pretrained=args.model, 
                     diameter=args.d, flow=args.o, cellprob=args.cellprob, 
                     min_size=args.min_size, min_ave_intensity=args.min_avgintensity, min_total_intensity=args.min_totalintensity, 
                     plot=args.s, output=args.r, channels=channels) 
        
        
    elif args.action=='cellphenotyping':        
        # remove [] and make a list
        files = args.f[1:-1].split(",")         
        positives = args.p[1:-1].split(",") 
        cutoffs = args.c[1:-1].split(",") 
        methods = args.m[1:-1].split(",")             
        marker_names = args.n[1:-1].split(",")
        channels = [int(i) for i in args.l[1:-1].split(",")]
        
        StainingAnalysis(files=files, marker_names=marker_names, positives=[p=='True' for p in positives], cutoffs=[float(c) for c in cutoffs], 
                         channels=channels, methods=methods, plot=True, output=args.r)
        
        
    elif args.action=='intensityanalysis':        
        # remove [] and make a list
        files = args.f[1:-1].split(",")         
        channels = [int(i) for i in args.l[1:-1].split(",")]
        
        IntensityAnalysis(files=files, channels=channels)
        
        
    elif args.action=='maskfile2outline':
        filename, extension = os.path.splitext(args.maskfile)
        if extension:
            maskfile2outline(args.maskfile)
        else:
            for i in os.listdir(args.maskfile):
                maskfile2outline(args.maskfile+"/"+i)
                
                
    elif args.action=="roifiles2mask":
        roifiles2mask (args.roifolder+"/*", args.width, args.height)


    elif args.action=="alignimages":
        if args.l == None: 
            sys.exit("ERROR: --l is required")            
        channels = [int(i)-1 for i in args.l[1:-1].split(",")]
        doalign (args.ref_image, args.image2, channels)


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
    
