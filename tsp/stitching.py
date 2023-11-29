from PIL import Image
import cv2, os, glob


def dostitch (config, directory):
    nrows=config['nrows']
    ncols=config['ncols']
    right_margin_overlap = config['rightMarginOverlap']
    top_margin_overlap = config['topMarginOverlap']
    panels = config['panels']
    
    # assume all position files have the same dimention
    filename = os.listdir(directory)[0] # e.g., M872956_Position1_CD3-BUV395.tiff
    im = cv2.imread(directory+filename, cv2.IMREAD_UNCHANGED)
    height, width = im.shape[:2]
    subjectid=filename.split('_')[0]
    marker= os.path.splitext(filename)[0].split('_')[2]
    
    width_reduced = int(width * (1-right_margin_overlap))
    height_reduced = int(height * (1-top_margin_overlap))
    
    # New image width and height
    new_width = width_reduced * ncols
    new_height = height_reduced * nrows
    
    # Create a new image with the appropriate size
    new_image = Image.new('RGB', (new_width, new_height))
    for i in range(nrows):
        for j in range(ncols):
            position = panels[i][j]
            if position=='empty':
                im = Image.new('RGB', (width_reduced, height_reduced))
            else:
                pattern = f'{subjectid}_{position}_*'            
                im = Image.open(glob.glob(directory + pattern)[0]).crop((0, height-height_reduced, width_reduced, height))
                
            # Compute the position where the next image should be pasted
            x_offset = width_reduced * j
            y_offset = height_reduced * i
            
            # Paste the image at the computed position
            new_image.paste(im, (x_offset, y_offset))
    
    # Save the new image
    new_image.save(f'{subjectid}_wholeslide_{marker}.png')
