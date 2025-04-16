from inference.Detection import Detector
from inference.Classification import Classificator
from inference.Inference import Inference

from PIL import Image

def main():
    detector = Detector(0, "runs/detect/train16/weights/best.pt")
    classificator = Classificator(0, "runs/classify/train3/weights/best.pt")
    inference = Inference(detector, classificator, 0.2)
    
    im = "C:\\Users\\maksi\\Documents\\dataset_full_images\\imgs\\26.jpeg"
    result = inference.forward(
        Image.open(im)
    )
    result.plot(show=True, font_size=10, line_width=2)

    result = detector.yolo.predict(source=im)
    result[0].plot(show=True, font_size=10, line_width=2)
    


if __name__ == "__main__":
    main()
