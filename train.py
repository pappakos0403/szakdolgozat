from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt") 
    results = model.train(data="C:/Users/Ákos/Desktop/Ákos/Szakdolgozat/football_analysis/training/football-players-detection-1/data.yaml",
                          epochs=100, imgsz=640, batch=4, device="cuda") 
