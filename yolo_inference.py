from ultralytics import YOLO
 
model = YOLO('./best.pt')

results = model.predict('input_videos/08fd33_4.mp4', save=True, project='output', name='yolo_results')
print(results[0])
print('=============')
for box in results[0].boxes:
    print(box)