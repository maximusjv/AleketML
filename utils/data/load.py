import os
from PIL import Image

def load_split_list(split_path):
    """Load image paths from a split file."""
    with open(split_path, "r") as f:
        return [
            os.path.normpath(
                os.path.join(os.path.dirname(split_path), line.strip())
            )
            for line in f
            if line.strip()
        ]
        
def load_image_files(source: str | list[str], yolo_dir = False):
    if isinstance(source, str):
        if os.path.isdir(source):
            image_dir = os.path.join(source, "images") if yolo_dir else source
            image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.jpeg'))]
        
        else:
            image_files = load_split_list(source)
    else:
        image_files = source
    
    return image_files

def load_yolo(source: str | list) -> dict:
    
    image_files = load_image_files(source, True)
        
    annotations = {}
    
    for image_file in image_files:
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        
        label_dir = os.path.normpath(os.path.join(os.path.dirname(image_file), '..', 'labels'))
        label_file = os.path.join(label_dir, f"{image_name}.txt")
        # Load image to get dimensions
        image = Image.open(image_file)
        img_width, img_height = image.size
     
        annotations[image_name] = []
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cat = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            # Convert YOLO format back to absolute coordinates (x1, y1, x2, y2)
                            x1 = (x_center - width/2) * img_width
                            y1 = (y_center - height/2) * img_height
                            x2 = (x_center + width/2) * img_width
                            y2 = (y_center + height/2) * img_height
                            
                            annotations[image_name].append([x1, y1, x2, y2, cat])
                            
    return annotations
    

        

def loader(source: str | list):
    image_list = load_image_files(source)
    for image_path in image_list:
        image_name =   image_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path)
        yield image_name, image                  
    