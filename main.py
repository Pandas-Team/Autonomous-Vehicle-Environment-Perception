from elements.yolo import YOLO, YOLO_Sign
from elements.PINet import LaneDetection
from elements.SGD import Inference
from elements.Curvlane import CurveLane
from elements.asset import cityscape_xyz, kitti_xyz, apply_mask
from utils.plots import plot_one_box
import matplotlib.pyplot as plt
from elements.asset import horiz_lines, detect_lines
import numpy as np
import cv2
from time import time as t
import datetime
import random
import sys
from SGDepth.arguments import InferenceEvaluationArguments

opt = InferenceEvaluationArguments().parse()


if opt.noshow and not opt.save:
    print("You're not getting any outputs!!\nExit")
    sys.exit()


detector = YOLO(opt.weights_detector)

if opt.lane_detector_type == 'culane':
    lane_detector = LaneDetection(opt.culane_model)
    print("CULane model loaded!")
if opt.lane_detector_type == 'curvelane':
    lane_detector = CurveLane(opt.curvelane_model)
    print("Curvelane model loaded!)

disparity_detector = Inference(opt.disp_detector)
sign_detector = YOLO_Sign(opt.weights_sign)

#Video Writer
cap = cv2.VideoCapture(opt.video)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# print(frame_count)

if opt.save:
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    out = cv2.VideoWriter(opt.output_name,  
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            opt.outputfps, (int(h), int(w)))

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

frame_num = 0
frame_drop = 30
while(cap.isOpened()):
    
    ret, frame = cap.read()
    frame_num += 1 
    if not frame_num% opt.frame_drop ==0:
        continue

    if ret:
        t1 = t() #Start Time
        # if opt.rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        main_frame = frame.copy()
        yoloOutput = detector.detect(frame)
        signOutput = sign_detector.detect_sign(frame)
        frame = lane_detector.Testing(frame)        
        disparity, seg_img = disparity_detector.inference(frame)


        frame = apply_mask(frame, seg_img)

        for obj in yoloOutput:
            coloraaaa = [255,0,0]
            xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
            
            if obj['label'] =='car' or obj['label'] == 'truck' or obj['label'] == 'bus':
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

        for sign in signOutput:
            xyxy = [sign['bbox'][0][0], sign['bbox'][0][1], sign['bbox'][1][0], sign['bbox'][1][1]]
            plot_one_box(xyxy, frame, label=sign['label'], color=colors_signs[sign['cls']], line_thickness=3)
        
        t2 = t() #End of frame time
        fps = np.round(1 / (t2-t1) , 3)   #Running FPS
        s = "FPS : "+ str(fps)
        if opt.fps:
            cv2.putText(frame, s, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness= 2)
        
        #Cross Walk Lines
        frame = horiz_lines(main_frame, frame)
        # Saving the output
        if opt.save:
            out.write(frame)
        
        if not opt.noshow:
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

    sys.stdout.write(
          "\r[Input Video : %s] [%d/%d Frames Processed] [Saving %s] [Show %s] [FPS : %f]"
          % (
              opt.video,
              frame_num,
              frame_count,
              opt.save,
              not  opt.noshow,
              fps
          )
      )
    
cap.release()
if not opt.noshow:
    cv2.destroyAllWindows()
