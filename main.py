from elements.yolo import YOLO, YOLO_Sign
from elements.PINet import LaneDetection
from elements.SGD import Inference
from elements.asset import cityscape_xyz, kitti_xyz, apply_mask
from utils.plots import plot_one_box
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import datetime
import random

detector = YOLO('weights/yolov5m.pt')
lane_detector = LaneDetection('weights/296_tensor(1.6947)_lane_detection_network.pkl')
disparity_detector = Inference('weights/model_full.pth')
detector_sign = YOLO_Sign('weights/Best_Sign_Model_TV.pt')

#Video Writer
cap = cv2.VideoCapture("/root/Documents/LaneATT/PINet_new/CULane/test.mov")

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

out = cv2.VideoWriter('./filename.mov',  
                        cv2.VideoWriter_fourcc(*'mp4v'), 
                        15, (int(h), int(w)))

# names = ['person', 'car', 'bus', 'truck', 'traffic light']
names = {
        'person': 0,
        'car' : 1,
        'bus': 2,
        'truck' : 3,
        'traffic light' : 4}
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

signs = ['Taghadom', 'Chap Mamnoo', 'Rast Mamnoo', 'SL30', 'Tavaghof Mamnoo',
         'Vorood Mamnoo', 'Mostaghom', 'SL40', 'SL50', 'SL60', 'SL70', 'SL80', 'SL100', 'No U-Turn']
colors_signs = [[random.randint(0, 255) for _ in range(3)] for _ in signs]


while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        yoloOutput = detector.detect(frame)
        sinOutput = detector_sign.detect_sign(frame)
        frame = lane_detector.Testing(frame)        
        disparity, seg_img = disparity_detector.inference(frame)

        try:
            frame = apply_mask(frame, seg_img)
        except:
            pass

        # plotting Yolo Ouput
        for obj in yoloOutput:
            coloraaaa = [255,0,0]
            xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
            
            if obj['label'] =='car' or obj['label'] == 'truck':
                x_pts = (obj['bbox'][0][0]+obj['bbox'][1][0])/2
                y_pts = (obj['bbox'][0][1]+obj['bbox'][1][1])/2

                Ry = 192/720
                Rx = 640/1280
                x_new, y_new =(Rx * x_pts, Ry * y_pts)

                cropped_disp = np.array(disparity[int(y_new-5): int(y_new+5), int(x_new-5): int(x_new+5)])
                sorted_value = np.sort(np.ravel(cropped_disp))
                sorted_value = np.array(sorted_value[int(0.7*len(sorted_value))-1:-1])
                mean = np.mean(sorted_value)

                x, y, z, distance = kitti_xyz(mean, x_new, y_new)

                if distance < 10:
                    plot_one_box(xyxy, frame, distance, label=obj['label'], color=colors[names[obj['label']]], line_thickness=3)
                else:
                    plot_one_box(xyxy, frame, label=obj['label'], color=colors[names[obj['label']]], line_thickness=3)
            else:
                plot_one_box(xyxy, frame, label=obj['label'], color=colors[names[obj['label']]], line_thickness=3)

        for sign in sinOutput:
            xyxy = [sign['bbox'][0][0], sign['bbox'][0][1], sign['bbox'][1][0], sign['bbox'][1][1]]
            plot_one_box(xyxy, frame, label=obj['label'], color=colors_signs[obj['cls']], line_thickness=3)
        
        out.write(frame)
        
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()
