import torch
import cv2
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = {0: 'person',
           2: 'car',
           5: 'bus',
           7: 'truck',
           9: 'traffic light',
           11: 'stop sign'}
margin = 0

class YOLO():
    def __init__(self,model_path):
        self.yolo_model = attempt_load(weights=model_path, map_location=device)
        print("Yolo model loaded!")
        self.conf_thres = 0.75
        self.iou_thres = 0.7

    def detect(self,left):
        """
            Input :
                    BGR image
            
                    
            Output:
            yolo return list of dict in format:
                {   label   :  str
                    bbox    :  [(xmin,ymin),(xmax,ymax)]
                    score   :  float
                    cls     :  int
                    }
        """
        img = cv2.resize(left, (640,384))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img,-1,0)
        img = torch.from_numpy(img).to(device)
        img = img.float()/255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.yolo_model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=None)
        items = []
        
        if pred[0] is not None and len(pred):
            for p in pred[0]:
                if int(p[5]) in list(classes.keys()): 
                    score = np.round(p[4].cpu().detach().numpy(),2)
                    label = classes[int(p[5])]
                    # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    xmin = int(p[0] * left.shape[1] /640)
                    ymin = int(p[1] * left.shape[0] /384)
                    xmax = int(p[2] * left.shape[1] /640)
                    ymax = int(p[3] * left.shape[0] /384)
                    xmin = xmin - margin if xmin - margin > 0 else 0
                    ymin = ymin - margin if ymin - margin > 0 else 0
                    xmax = xmax + margin if xmax + margin < left.shape[1] else left.shape[1]
                    ymax = ymax + margin if ymax + margin < left.shape[0] else left.shape[0]

                    item = {'label': label,
                            'bbox' : [(xmin,ymin),(xmax,ymax)],
                            'score': score,
                            'cls' : int(p[5])
                            }

                    items.append(item)

        return(items)


classes_sign = {0: 'Taghadom',
                1: 'Chap Mamnoo',
                2: 'Rast Mamnoo',
                3: 'SL30',
                4: 'Tavaghof Mamnoo',
                5: 'Vorood Mamnoo',
                6: 'Mostaghom',
                7: 'SL40',
                8: 'SL50',
                9: 'SL60',
                10: 'SL70',
                11: 'SL80',
                12: 'SL100',
                13: 'No U-Turn',
                }

margin_sign = 0

class YOLO_Sign():
    def __init__(self,model_path):
        self.yolo_model = attempt_load(weights=model_path, map_location=device)
        print("Sign Detection model loaded!")
        self.conf_thres = 0.75
        self.iou_thres = 0.7

    def detect_sign(self,left):
        """
            Input :
                    BGR image
            
                    
            Output:
            yolo return list of dict in format:
                {   label   :  str
                    bbox    :  [(xmin,ymin),(xmax,ymax)]
                    score   :  float
                    cls     :  int
                    }
        """
        img = cv2.resize(left, (640,384))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img,-1,0)
        img = torch.from_numpy(img).to(device)
        img = img.float()/255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.yolo_model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres= self.conf_thres, iou_thres=self.iou_thres, classes=None)
        items = []
        
        if pred[0] is not None and len(pred):
            for p in pred[0]:
                score = np.round(p[4].cpu().detach().numpy(),2)
                label = classes_sign[int(p[5])]
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                xmin = int(p[0] * left.shape[1] /640)
                ymin = int(p[1] * left.shape[0] /384)
                xmax = int(p[2] * left.shape[1] /640)
                ymax = int(p[3] * left.shape[0] /384)
                xmin = xmin - margin_sign if xmin - margin_sign > 0 else 0
                ymin = ymin - margin_sign if ymin - margin_sign > 0 else 0
                xmax = xmax + margin_sign if xmax + margin_sign < left.shape[1] else left.shape[1]
                ymax = ymax + margin_sign if ymax + margin_sign < left.shape[0] else left.shape[0]

                item = {'label': label,
                        'bbox' : [(xmin,ymin),(xmax,ymax)],
                        'score': score,
                        'cls': int(p[5])
                        }

                items.append(item)

        return(items)