# AleketML
### Sunflower broomrape nodules' detection model
#### By Maksim Vinokur, Dor Oppenheim, Dana Siso

# Object Detection Inference Tool (infer.py)
This tool supports batch processing, image patching, and provides detailed statistics and visualizations of detected objects.

## Installation

>Note: you should have python3 installed

```bash
git clone https://github.com/maximusjv/AleketML.git
cd AleketML
pip install -r requirements.txt
```

For better performance it's highly recommended to have gpu compatible with ndivida CUDA, and have cuda installed.

For more details: https://developer.nvidia.com/cuda-toolkit

## Usage

```bash
python infer.py <model_path> <images_path> [options]
```

### Required Arguments

- `model_path`: Path to the trained model weights (FasterRCNN ResNet50 FPN V2) (you can use default pretrained "model.pth")
- `images_path`: Path to either:
  - A directory containing images (supported formats: .jpeg, .jpg, .png) (other pillow supported image formats can be added manually)
  - A text file containing a list of image paths

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --output_dir | "output" | Directory to save results |
| --iou_thresh | 0.2 | IoU threshold for non-maximum suppression |
| --score_thresh | 0.8 | Confidence score threshold for detections |
| --num_of_annotations_to_save | 0 | Number of annotated images to save (-1 for all) |
| --save_annotated_images | False | Save images with visualized detections |
| --use_merge | False | Use WBF (Weighted Box Fusion) for post-processing |
| --images_per_batch | 1 | Number of images to process in each batch |
| --image_size_factor | 1.0 | Factor to resize input images |
| --detections_per_image | 300 | Maximum number of detections per image |
| --detections_per_patch | 100 | Maximum number of detections per patch |
| --patches_per_batch | 4 | Number of patches to process in each batch |
| --patch_size | 1024 | Size of each image patch |
| --patch_overlap | 0.2 | Overlap between adjacent patches |

## Output

The tool generates the following outputs in the specified output directory:

### 1. Statistics File (stats.csv)
Contains per-image statistics including:
- Area of detected objects per class
- Count of detected objects per class

### 2. Bounding Box Files (if enabled)
For each processed image, a CSV file containing:
- xmin, ymin, xmax, ymax coordinates
- Class name for each detection

### 3. Annotated Images (if enabled)
Visualizations of the original images with:
- Drawn bounding boxes
- Class labels for detected objects

## Example Usage

Basic usage with default parameters:
```bash
python infer.py models/faster_rcnn.pth images/
```

Advanced usage with custom parameters:
```bash
python infer.py models/faster_rcnn.pth images/ \
    --output_dir results \
    --iou_thresh 0.6 \
    --score_thresh 0.7 \
    --num_of_annotations_to_save 10 \
    --save_annotated_images \
    --use_merge \
    --images_per_batch 4 \
    --patch_size 512 \
    --patch_overlap 0.3
```

## Notes

- Area calculations use elliptical approximation (π * width/2 * height/2) for each bounding box predicted.
- You also can finetune your own model by using the train.ipynb

## License
Currently no license, but I will change it