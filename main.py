from elements.yolo import YOLO, YOLO_Sign
from elements.PINet import LaneDetection
from elements.SGD import SGDepth_Model
from elements.light_classifier import light_classifier
from elements.asset import plot_object_colors, depth_estimator, apply_mask, apply_all_mask, ROI, plot_one_box, ui, horiz_lines
import numpy as np
import os
import cv2
from time import time as t
import sys
from datetime import timedelta
from SGDepth.arguments import InferenceEvaluationArguments


opt = InferenceEvaluationArguments().parse()


if opt.noshow and not opt.save:
    print("You're not getting any outputs!!\nExit")
    sys.exit()

# Load Models
detector = YOLO() 
sign_detector = YOLO_Sign(opt.weights_sign) 
light_detector = light_classifier(opt.weights_light) 
lane_detector = LaneDetection(opt.culane_model)
depth_seg_estimator = SGDepth_Model(opt.disp_detector)


# Video Writer
cap = cv2.VideoCapture(opt.video)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

resize = not ((w == 1280) and (h == 720))


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
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 280
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter(output_video_name,  
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            opt.outputfps, (int(w), int(h)))

# Create color palettes for visualization 
obj_colors, sign_colors = plot_object_colors()

frame_num = 0

total_fps = []
while(cap.isOpened()):
    
    ret, frame = cap.read()
    frame_num += 1
    if not frame_num % opt.frame_drop == 0:
        continue

    if ret:
        tc = t() # Start Time
        
        if resize:
            frame = cv2.resize(frame , (int(1280),int(720)))

        main_frame = frame.copy()
        yoloOutput = detector.detect(frame)  
        signOutput = sign_detector.detect_sign(frame)
        depth, seg_img = depth_seg_estimator.inference(frame)


        # # Dynamic ROI Generation
        masked_image = ROI(main_frame, seg_img)


        # ### Sidewalk detection ### 
        if opt.mode != 'night':
            frame = apply_mask(frame, seg_img, mode = opt.mode)


        # ### Lane Detection ###
        frame = lane_detector.detect_lane(frame, masked_image)


        # ### Object Detection ###
        depth_values = []
        for obj in yoloOutput.values:
            xyxy = [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])] # [Xmin, Ymin, Xmax, Ymax]

            if obj[-2] in ['traffic light', 'stop sign', 'pedestrian'] :
                plot_one_box(xyxy, frame, label=obj[-2], color=obj_colors[obj[-2]], line_thickness=3)
            else:
                ### Distance Measurement ###
                obj_area = (xyxy[3] - xyxy[1]) * (xyxy[2] - xyxy[0])
                if obj_area > 6000:
                    depth_value = depth_estimator(xyxy, depth=depth, seg=seg_img, obj_name=obj[-1], mask_state=False)
                    depth_values.append(depth_value)
                    plot_one_box(xyxy, frame, distance=depth_value, label=obj[-2], color=obj_colors[obj[-2]], line_thickness=3)
                else:
                    plot_one_box(xyxy, frame, label=obj[-2], color=obj_colors[obj[-2]], line_thickness=3)


        ### Sign Detection ###
        for sign in signOutput.values:
            xyxy = [sign[0], sign[1], sign[2], sign[3]]
            plot_one_box(xyxy, frame, label=sign[-1],  color=sign_colors[sign[-1]], line_thickness=3)

        # ### Cross Walk Lines ###
        # # frame = horiz_lines(main_frame, frame, mode = opt.mode)
        
        ### UI ###
        ui_bg = ui(main_frame, yoloOutput, light_detector, signOutput, depth_values)
        frame = cv2.hconcat([frame, ui_bg])


        t2 = t() # End of frame time
        fps = (1/(t2-tc))
        avg_fps = np.round(fps , 3)
        estimated_time = (frame_count - frame_num) / avg_fps
        estimated_time = str(timedelta(seconds=estimated_time)).split('.')[0]
        s = "FPS : "+ str(fps)
        if opt.fps:
            cv2.putText(frame, s, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness= 2)
        

        # Saving the output
        if opt.save:
            out.write(frame)
            if opt.save_frames:
                cv2.imwrite(os.path.join(output_frames_folder , '{0:04d}.jpg'.format(int(frame_num))) , frame)
        

        if not opt.noshow:
            cv2.imshow('frame', frame)
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
