#############################################################################################################
##
##  Source code for testing
##
#############################################################################################################

import cv2
import json
import torch
import agent
import numpy as np
from copy import deepcopy
from data_loader import Generator
import time
from parameters import Parameters
import util
import os
import csaps

p = Parameters()

class LaneDetection():
    def __init__(self,model_path):
        self.yolo_model = attempt_load(weights=model_path, map_location=device)
def Testing():
    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.model_path == "":
        lane_agent = agent.Agent()
    else:
        lane_agent = agent.Agent()
        lane_agent.load_weights(296, "tensor(1.6947)")
	
    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()

    ##############################
    ## testing
    ##############################

    elif p.mode == 1: # check model with video
        cap = cv2.VideoCapture("/root/Documents/LaneATT/PINet_new/CULane/test.mov")

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


        out = cv2.VideoWriter('./test_result/filename.mov',  
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                15, (int(h), int(w)))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # torch.cuda.synchronize()
                prevTime = time.time()
                frame = cv2.resize(frame, (512,256))/255.0
                frame = np.rollaxis(frame, axis=2, start=0)
                _, _, ti = test(lane_agent, np.array([frame])) 
                curTime = time.time()
                sec = curTime - prevTime
                fps = 1/(sec)
                s = "FPS : "+ str(fps)
                ti[0] = cv2.resize(ti[0], (1280,720))
                cv2.putText(ti[0], s, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                
                # out.write(ti[0])
                
                cv2.imshow('frame',ti[0])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
            
        cap.release()
        cv2.destroyAllWindows()

############################################################################
## test on the input test image
############################################################################
def test(lane_agent, test_images, thresh = p.threshold_point, index= -1):

    result = lane_agent.predict_lanes_test(test_images)
    # torch.cuda.synchronize()
    confidences, offsets, instances = result[index]
    
    num_batch = len(test_images)

    out_x = []
    out_y = []
    out_images = []
    
    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i])
        image =  np.rollaxis(image, axis=2, start=0)
        image =  np.rollaxis(image, axis=2, start=0)*255.0
        image = image.astype(np.uint8).copy()

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()

        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)
        
        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
                
        # sort points along y 
        in_x, in_y = util.sort_along_y(in_x, in_y)  

        result_image = util.draw_points(in_x, in_y, deepcopy(image))

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)

    return out_x, out_y,  out_images

############################################################################
## eliminate result that has fewer points than threshold
############################################################################
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>5:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

############################################################################
## generate raw output
############################################################################
def generate_result(confidance, offsets,instance, thresh):

    mask = confidance > thresh

    grid = p.grid_location[mask]
    offset = offsets[mask]
    feature = instance[mask]

    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                flag = 0
                index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= p.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                elif len(lane_feature) < 12:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                
    return x, y

if __name__ == '__main__':
    Testing()
