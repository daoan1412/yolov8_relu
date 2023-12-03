from ultralytics import YOLO
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


model = YOLO('/home/daoan/Projects/yolov8_relu/ultralytics/cfg/models/v8/yolov8s-pose.yaml')

results = model.train(data='my-coco-pose.yaml', epochs=200, imgsz=640, batch=64)

