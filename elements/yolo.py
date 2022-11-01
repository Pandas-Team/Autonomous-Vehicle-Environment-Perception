import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class YOLO():
    def __init__(self):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', _verbose=False).to(device)
        self.yolo_model.conf = 0.55  # confidence threshold (0-1)
        self.yolo_model.iou = 0.7  # NMS IoU threshold (0-1)
        self.yolo_model.classes = [0, 2, 5, 7, 9, 11]  
        print('Yolov5 model loaded!')

    def detect(self, img):
        results = self.yolo_model(img[...,::-1], size=640)

        items = results.pandas().xyxy[0]
        items['labels'] = items['name'].copy()
        items['name'] = items['name'].replace(['car', 'truck', 'bus'],'vehicle')
        items['name'] = items['name'].replace( 'person', 'pedestrian' )

        return items

class YOLO_Sign():
    def __init__(self, model_path):
        self.yolo_model = torch.hub.load('yolov5', 'custom', path=model_path, source='local').to(device)
        self.yolo_model.conf = 0.82  # confidence threshold (0-1)
        self.yolo_model.iou = 0.75  # NMS IoU threshold (0-1)
        print('Sign model loaded!')

    def detect_sign(self, img):
        results = self.yolo_model(img[...,::-1], size=640)

        items = results.pandas().xyxy[0]

        return items




