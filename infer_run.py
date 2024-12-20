import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Run inference and generate statistics."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model weights (fasterrcnn_resnet50_fpn_v2)",
    )
    parser.add_argument(
        "images_path",
        type=str,
        help="path to image dir or a file contatining list of images to infer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="output directory (default: output)",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.25,
        help="IOU threshold for postproccessing (default: 0.25)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.75,
        help="score threshold for object detection (default: 0.75)",
    )
    parser.add_argument(
        "--save_annots", action="store_true", help="save annotations",
    )
    parser.add_argument(
        "--save_images", action="store_true", help="save annotated images"
    )
    parser.add_argument(
        "--image_size_factor",
        type=float,
        default=1.0,
        help="factor to resize input images (default: 1.0)",
    )
    parser.add_argument(
        "--pre_wbf_detections",
        type=int,
        default=2500,
        help="maximum number of detections per image before wbf postprocess (default: 2500)",
    )
    parser.add_argument(
        "--detections_per_patch",
        type=int,
        default=3000,
        help="maximum number of detections per patch (default: 300)",
    )
    parser.add_argument(
        "--patches_per_batch",
        type=int,
        default=4,
        help="number of patches to process in a batch (default: 4)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=1024,
        help="size of each image patch (default: 1024)",
    )
    parser.add_argument(
        "--patch_overlap",
        type=float,
        default=0.2,
        help="overlap between adjacent patches (default: 0.2)",
    )

    args = parser.parse_args()

    print("LOADING LIBRARIES...")
    
    import torch
    import torchvision
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"******USING DEVICE: {device}******")
    
    from utils.infer import load_model, load_pathes, infer
    from utils.consts import NUM_TO_CLASSES
    from utils.predictor import Predictor
    
    model = load_model(args.model_path, device)
    pathes = load_pathes(args.images_path)
    
    print(f"******MODEL LOADED******")   
    
    predictor = Predictor(
        model,
        device,
        image_size_factor=args.image_size_factor,
        pre_wbf_detections=args.pre_wbf_detections,
        detections_per_patch=args.detections_per_patch,
        patches_per_batch=args.patches_per_batch,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
    )
    infer(
        predictor,
        pathes,
        NUM_TO_CLASSES,
        args.output_dir,
        args.iou_thresh,
        args.score_thresh,
        args.save_annots,
        args.save_images,
        verbose=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Sorry something went wrong! :(")
        print(e.with_traceback(e.__traceback__))
