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
        default=0.2,
        help="IOU threshold for postproccessing (default: 0.2)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.8,
        help="score threshold for object detection (default: 0.8)",
    )
    parser.add_argument(
        "--num_of_annotations_to_save",
        type=int,
        default=0,
        help="number of annotated images to save (default: 0, -1 for all)",
    )
    parser.add_argument(
        "--save_annotated_images", action="store_true", help="save annotated images"
    )
    parser.add_argument(
        "--image_size_factor",
        type=float,
        default=1.0,
        help="factor to resize input images (default: 1.0)",
    )
    parser.add_argument(
        "--detections_per_image",
        type=int,
        default=500,
        help="maximum number of detections per image (default: 300)",
    )
    parser.add_argument(
        "--detections_per_patch",
        type=int,
        default=100,
        help="maximum number of detections per patch (default: 100)",
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
        detections_per_image=args.detections_per_image,
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
        args.num_of_annotations_to_save,
        args.save_annotated_images,
        verbose=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Sorry something went wrong! :(")
        raise e
        print(e.with_traceback(e.__traceback__))
