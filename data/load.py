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
        annotations[image_name] = load_yolo_labels(image_file)
                            
    return annotations
    
def load_yolo_labels(image_file):
        """Retrieves annotations for the specified image file."""
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        label_dir = os.path.normpath(os.path.join(os.path.dirname(image_file), '..', 'labels'))
        label_file = os.path.join(label_dir, f"{image_name}.txt")
        
        image = Image.open(image_file)
        img_width, img_height = image.size
        image.close()
        
        annotations = []
        
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
                            
                            annotations.append([x1, y1, x2, y2, cat])
        
        return annotations
    
def load_as_coco(source: str | list, classes: dict = {}):
    """Converts a custom dataset to COCO API format.
    Args:
        dataset: The custom dataset to convert.

    Returns:
        A COCO dataset object.
    """
    image_files = load_image_files(source, True)
    
    coco_api_dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    ann_id = 1
    img_id = 0
    names_to_ids = {}

    for image_file in image_files:
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        
        label_dir = os.path.normpath(os.path.join(os.path.dirname(image_file), '..', 'labels'))
        label_file = os.path.join(label_dir, f"{image_name}.txt")
        # Load image to get dimensions
        image = Image.open(image_file)
        img_width, img_height = image.size
        
        img_id += 1
        names_to_ids[image_name] = img_id
        img_entry = {"id": img_id, "height": img_height, "width": img_width}
        coco_api_dataset["images"].append(img_entry)
        
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
                            
                        
                            w = x2-x1
                            h = y2-y1
                                                    
                            ann = {
                                "image_id": img_id,
                                "bbox": [x1, y1, w, h],
                                "category_id": cat,
                                "area": w * h,
                                "iscrowd": 0,
                                "id": ann_id,
                            }
            
                            categories.add(cat)
                            coco_api_dataset["annotations"].append(ann)
                            ann_id += 1

            
    coco_api_dataset["categories"] = [
        {"id": i, "label": classes.get(i, i)} for i in sorted(categories)
    ] 
    
    return coco_api_dataset, names_to_ids
        

class loader:
    def __init__(self, source: str | list):
        self.image_list = load_image_files(source)
    
    def __iter__(self):
        for image_path in self.image_list:
            image_name =   image_name = os.path.splitext(os.path.basename(image_path))[0]
            image = Image.open(image_path)
            yield image_name, image  
            
    def __len__(self):
        return len(self.image_list)
                    
    