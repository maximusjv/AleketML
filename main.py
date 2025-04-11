from inference_pipeline.Detection import Detection

def main():
    detector = Detection("yolo11n.pt")
    im = "C:\\Users\\maksi\\Documents\\AleketML\\venv\\Lib\\site-packages\\ultralytics\\assets\\zidane.jpg"
    result = detector.forward(
        im
    )
    result.plot(show=True, pil=True)

    result = detector.yolo.predict(source=im)
    result[0].plot(show=True)
    
    print(result)


if __name__ == "__main__":
    main()
