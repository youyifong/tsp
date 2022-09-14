import argparse, glob
import numpy as np
from syotil import *

def main():
    
    parser = argparse.ArgumentParser(description='syotil parameters')
    parser.add_argument('--action', type=str, help='maskfile2outline or checkprediction')
    parser.add_argument('--name', 
                        type=str, help='maskfile or prediction folder', required=False)
    parser.add_argument('--metric', 
                        default='csi', type=str, help='csi or bias or tpfpfn or coloring', required=False)    
    parser.add_argument('--verbose', action='store_true', help='show information about running and settings and save to log')    
    args = parser.parse_args()


    if args.action=='maskfile2outline':
        #for i in range(len(args.maskfiles)):
        maskfile2outline(args.name)

    elif args.action=='checkprediction':
        pred_name = sorted(glob.glob(args.name+'/*_masks.png')) 
        
        thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
        res_mat = []
        for i in range(len(pred_name)):
            y_pred = imread(pred_name[i])
            
            filename = pred_name[i].split('/')[-1]
            filename = filename.split('_img_cp_masks.png')[0]
            filename = '../testmasks/' + filename + '_masks.png'
            
            labels = imread(filename)
            
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
                color_fp_fn(masks_name[i], pred_name[i])
                        
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
    
