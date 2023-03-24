nohup python -m tsp runcellpose --f DD_Les_CD15_stitched.png --l [3,0] --model cytotrain7 &

python -m tsp dist2boundary --cells DD_Les_CD15_stitched_masks.csv --boundaryroi DD_Les_skin_boundary.roi

python -m tsp regionmembership --cells DD_Les_CD15_stitched_masks_d2b.csv --regionroi [DD_Les_normal_side.roi,DD_Les_lesion_side.roi]
