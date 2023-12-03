from ultralytics import YOLO

model = YOLO('/home/daoan/Projects/yolov8_relu/runs/pose/train13/weights/best.pt')  # load a custom trained model

model.export(format='onnx', opset=11)