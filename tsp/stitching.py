from PIL import Image
import cv2, os, glob
from tsp import imread


def dostitch (config, directory):
    # assume all position files have the same dimension
    filename = os.listdir(directory)[0]  # e.g., M872956_Position1_CD3-BUV395.tiff
    im = cv2.imread(directory + filename, cv2.IMREAD_UNCHANGED)
    height, width = im.shape[:2]
    subjectid = filename.split('_')[0]
    marker = os.path.splitext(filename)[0].split('_')[2]

    nrows=config['nrows']
    ncols=config['ncols']
    
    top_margin_overlap = config['topMarginOverlap']
    right_margin_overlap = config.get('rightMarginOverlap')
    left_margin_overlap = config.get('leftMarginOverlap')

    if right_margin_overlap is not None and left_margin_overlap is not None:
        exit("unexpected situation: both left_margin_overlap and right_margin_overlap are present")

    if right_margin_overlap is not None:
        # check if right_margin_overlap is a float between 0 and 1 or an integer
        if isinstance(right_margin_overlap, float) and 0 <= right_margin_overlap < 1:
            width_reduced = int(width * (1 - right_margin_overlap))
        elif isinstance(right_margin_overlap, int) and right_margin_overlap >= 0:
            width_reduced = width - right_margin_overlap
        else:
            exit("rightMarginOverlap should be a float between 0 and 1 or an integer >= 0")

    elif left_margin_overlap is not None:
        # check if left_margin_overlap is a float between 0 and 1 or an integer
        if isinstance(left_margin_overlap, float) and 0 <= left_margin_overlap < 1:
            width_reduced = int(width * (1 - left_margin_overlap))
        elif isinstance(left_margin_overlap, int) and left_margin_overlap >= 0:
            width_reduced = width - left_margin_overlap
        else:
            exit("leftMarginOverlap should be a float between 0 and 1 or an integer >= 0")

    else:
        exit("only one of rightMarginOverlap and rightMarginOverlap should be present")
    
    height_reduced = int(height * (1-top_margin_overlap))
    
    # New image width and height
    new_width = width_reduced * ncols
    new_height = height_reduced * nrows

    # find out whether the image is in color or grayscale
    if len(imread(glob.glob(directory + f'{subjectid}_*')[0]).shape)==2:
        mode='L'
    else:
        mode='RGB'

    # Create a new image with the appropriate size
    panels = config['panels']
    new_image = Image.new(mode, (new_width, new_height))
    for i in range(nrows):
        for j in range(ncols):
            position = panels[i][j]
            if position=='empty':
                im = Image.new(mode, (width_reduced, height_reduced))
            else:
                pattern = f'{subjectid}_{position}_*'
                if right_margin_overlap is not None:
                    im = Image.open(glob.glob(directory + pattern)[0]).crop((0, height-height_reduced, width_reduced, height))
                elif left_margin_overlap is not None:
                    im = Image.open(glob.glob(directory + pattern)[0]).crop((width-width_reduced, height-height_reduced, width, height))
                
            # Compute the position where the next image should be pasted
            x_offset = width_reduced * j
            y_offset = height_reduced * i
            
            # Paste the image at the computed position
            new_image.paste(im, (x_offset, y_offset))
    
    # Save the new image
    new_image.save(f'{subjectid}_wholeslide_{marker}.png')
