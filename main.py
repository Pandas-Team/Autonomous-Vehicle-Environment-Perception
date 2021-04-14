from elements.yolo import YOLO, YOLO_Sign
from elements.PINet import LaneDetection
from elements.SGD import Inference
from elements.asset import cityscape_xyz, kitti_xyz, apply_mask, ROI, kitti_xyz_dist, cityscape_xyz_dist, plot_one_box
from elements.asset import horiz_lines, detect_lines
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from time import time as t
import datetime
import random
import sys
from datetime import timedelta
from SGDepth.arguments import InferenceEvaluationArguments


opt = InferenceEvaluationArguments().parse()


if opt.noshow and not opt.save:
    print("You're not getting any outputs!!\nExit")
    sys.exit()


detector = YOLO(opt.weights_detector)

if opt.lane_detector_type == 'culane':
    lane_detector = LaneDetection(opt.culane_model, opt.lane_detector_type)
    print("CULane model loaded!")
if opt.lane_detector_type == 'curvelane':
    lane_detector = LaneDetection(opt.curvelane_model, opt.lane_detector_type)
    print("Curvelane model loaded!")

disparity_detector = Inference(opt.disp_detector)
sign_detector = YOLO_Sign(opt.weights_sign)

#Video Writer
cap = cv2.VideoCapture(opt.video)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

rotate = w<h
if rotate :
    h,w = w,h
resize = not ((w == 1280) and (h == 720))
print('resize ', resize)
print('rotate ', rotate)


if opt.save:
    if len(opt.output_name.split('.'))==1:
        opt.output_name += '.mp4'
    output_video_folder = os.path.join('outputs/', opt.output_name.split('.')[0])
    if opt.save_frames:
        output_frames_folder = os.path.join(output_video_folder, 'frames')
        os.makedirs(output_frames_folder, exist_ok=True)
    output_video_name = os.path.join(output_video_folder, opt.output_name)
    os.makedirs(output_video_folder, exist_ok = True)
    print(output_video_folder)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    out = cv2.VideoWriter(output_video_name,  
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            opt.outputfps, (int(h), int(w)))

names = {
        'person': 0,
        'car' : 1,
        'bus': 2,
        'truck' : 3,
        'traffic light' : 4,
        'stop sign' : 5}
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

signs = ['Taghadom', 'Chap Mamnoo', 'Rast Mamnoo', 'SL30', 'Tavaghof Mamnoo',
         'Vorood Mamnoo', 'Mostaghom', 'SL40', 'SL50', 'SL60', 'SL70', 'SL80', 'SL100', 'No U-Turn']
colors_signs = [[random.randint(0, 255) for _ in range(3)] for _ in signs]
avg_fps = 0 #Average FPS
frame_num = 0

while(cap.isOpened()):
    
    ret, frame = cap.read()
    frame_num += 1
    if not frame_num% opt.frame_drop ==0:
        continue

    if ret:
        t1 = t() #Start Time
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        if resize:
            frame = cv2.resize(frame , (int(1280),int(720)))

        main_frame = frame.copy()
        yoloOutput = detector.detect(frame)
        signOutput = sign_detector.detect_sign(frame)
        disparity, seg_img = disparity_detector.inference(frame)
        
        #set the desired area to eliminate bad distances
        masked_image = ROI(main_frame)
        frame = lane_detector.Testing(frame, masked_image)
        if opt.mode != 2:
            frame = apply_mask(frame, seg_img, masked_image)

        for obj in yoloOutput:
            xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
            depth = []
            if obj['label'] =='car' or obj['label'] == 'truck' or obj['label'] == 'bus':
                x_pts = (obj['bbox'][0][0]+obj['bbox'][1][0])/2
                y_pts = (obj['bbox'][0][1]+obj['bbox'][1][1])/2

                #ِDistance Measurement
                if np.dot(masked_image[int(y_pts), int(x_pts)], main_frame[int(y_pts), int(x_pts)]) != 0:
                    Ry = 192/720
                    Rx = 640/1280
                    x_new, y_new =(Rx * x_pts, Ry * y_pts)

                    cropped_img = main_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    cropped_disp = np.array(disparity[int(xyxy[1]*Ry):int(xyxy[3]*Ry), int(xyxy[0]*Rx):int(xyxy[2]*Rx)]) 
                    cropped_img = cv2.resize(cropped_img, (cropped_disp.shape[1], cropped_disp.shape[0]))
                    cropped_img = cropped_img[int(cropped_img.shape[0]/2 - 20): int(cropped_img.shape[0]/2 + 20),
                                            int(cropped_img.shape[1]/2 - 20): int(cropped_img.shape[1]/2 + 20)]

                    indices = np.where(cropped_img!= [0])
                    coordinates = zip(indices[0], indices[1])

                    for x,y in coordinates:
                        try:
                            depth.append([x, y, cropped_disp[y,x]])
                        except:
                            pass
                    
                    if opt.depth_mode == 'kitti':
                        distance = kitti_xyz_dist(depth)
                    else : 
                        distance = cityscape_xyz_dist(depth)

                    printed_distance = np.mean(np.array(sorted(distance)[:15]))

                    if  printed_distance < 10:
                        plot_one_box(xyxy, frame, printed_distance, label=obj['label'], color=colors[names[obj['label']]], line_thickness=3)
                    else:
                        plot_one_box(xyxy, frame, label=obj['label'], color=colors[names[obj['label']]], line_thickness=3)
                else:
                        plot_one_box(xyxy, frame, label=obj['label'], color=colors[names[obj['label']]], line_thickness=3)
            else:
                plot_one_box(xyxy, frame, label=obj['label'], color=colors[names[obj['label']]], line_thickness=3)

        for sign in signOutput:
            xyxy = [sign['bbox'][0][0], sign['bbox'][0][1], sign['bbox'][1][0], sign['bbox'][1][1]]
            plot_one_box(xyxy, frame, label=sign["label"],  color=colors_signs[sign['cls']], line_thickness=3)
        
        t2 = t() #End of frame time
        fps = np.round(1 / (t2-t1) , 3)   #Running FPS
        avg_fps = fps * 0.05 + 0.95 * avg_fps
        estimated_time = (frame_count - frame_num) / avg_fps
        estimated_time = str(timedelta(seconds=estimated_time)).split('.')[0]
        s = "FPS : "+ str(fps)
        if opt.fps:
            cv2.putText(frame, s, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness= 2)
        
        
        #Cross Walk Lines
        frame = horiz_lines(main_frame, frame, mode = opt.mode)


        # Saving the output
        if opt.save:
            out.write(frame)
            if opt.save_frames:
                cv2.imwrite(os.path.join(output_frames_folder , '{0:04d}.jpg'.format(int(frame_num))) , frame)
        

        if not opt.noshow:
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

    sys.stdout.write(
        "\r[Input Video : %s] [%d/%d Frames Processed] [FPS : %f] [ET : %s]"
        % (
            opt.video,
            frame_num,
            frame_count,
            fps,
            estimated_time
        )
    )
    
cap.release()

if not opt.noshow:
    cv2.destroyAllWindows()
