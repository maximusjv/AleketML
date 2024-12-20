# AleketML
### Sunflower broomrape nodules' detection model
#### By Maksim Vinokur, Dor Oppenheim, Dana Siso

# Object Detection Inference Tool (infer.py)
This tool supports batch processing, image patching, and provides detailed statistics and visualizations of detected objects.

## Installation

>Note: you should have python (version >=3.7 & <3.11) installed

```bash
git clone https://github.com/maximusjv/AleketML.git
cd AleketML
pip install -r requirements.txt
```

>It's recommended to use python virtual enviroment

For better performance it's highly recommended to have gpu compatible with ndivida CUDA, and have cuda installed.

For more details: https://developer.nvidia.com/cuda-toolkit

Trained model can be found on: [Google drive](https://drive.google.com/file/d/133G4ytupqJQYqZ6LV1Cjs0I1Rj1Vdrun/view?usp=drive_link)
## Usage

```bash
python infer_run.py <model_path> <images_path> [options]
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
| --iou_thresh | 0.25 | IoU threshold for non-maximum suppression |
| --score_thresh | 0.75 | Confidence score threshold for detections |
| --save_annots | False | Save detections annotations |
| --save_images | False | Save images with visualized detections |
| --image_size_factor | 1.0 | Factor to resize input images |
| --pre_wbf_detections | 2500 | Maximum number of detections per image before wbf postprocess  |
| --detections_per_patch | 300 | Maximum number of detections per patch |
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
python infer_run.py model.pth images/
```

Advanced usage with custom parameters:
```bash
python infer_run.py model.pth images/ \
    --output_dir results \
    --iou_thresh 0.25 \
    --score_thresh 0.75 \
    --save_annots \
    --save_images \
```

## Accuracy 

With default parameters, the tool showed an AP of 0.529, a Precision of 0.665, and a Recall of 0.625. If you are interested only in detecting the nodules and not classifying them as healthy or necrotic, these metrics increase to 0.647, 0.748, and 0.72, respectively.

## Notes

- Area calculations use elliptical approximation (π * width/2 * height/2) for each bounding box predicted.
- You also can finetune your own model by using the train.ipynb

## License

MIT License

Copyright (c) 2024 Maksim Vinokur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.