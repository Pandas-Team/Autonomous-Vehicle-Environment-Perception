
import torch
from CurveLanes.agent import *
from CurveLanes.parameters import Parameters
from CurveLanes.util import *
import cv2
import numpy as np
from copy import deepcopy
import time
import os

p = Parameters()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CurveLane():
    def __init__(self,model_path):
        self.lane_agent = Agent()
        self.lane_agent.load_weights(model_path)

    def Testing(self, frame):
        if torch.cuda.is_available():
            self.lane_agent.cuda()
        frame = cv2.resize(frame, (512,256))/255.0
        frame = np.rollaxis(frame, axis=2, start=0)
        _, _, ti = self.test(self.lane_agent, np.array([frame])) 
        ti[0] = cv2.resize(ti[0], (1280,720))

        return ti[0]

    def test(self, lane_agent, test_images, thresh = p.threshold_point, index= -1):

        result = lane_agent.predict_lanes_test(test_images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        confidences, offsets, instances = result[index]
        
        num_batch = len(test_images)

        out_x = []
        out_y = []
        out_images = []
        
        for i in range(num_batch):
            # test on test data set
            image = deepcopy(test_images[i])
            image = np.rollaxis(image, axis=2, start=0)
            image = np.rollaxis(image, axis=2, start=0)*255.0
            image = image.astype(np.uint8).copy()

            confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()

            offset = offsets[i].cpu().data.numpy()
            offset = np.rollaxis(offset, axis=2, start=0)
            offset = np.rollaxis(offset, axis=2, start=0)
            
            instance = instances[i].cpu().data.numpy()
            instance = np.rollaxis(instance, axis=2, start=0)
            instance = np.rollaxis(instance, axis=2, start=0)

            # generate point and cluster
            raw_x, raw_y = self.generate_result(confidence, offset, instance, thresh)

            # eliminate fewer points
            in_x, in_y = self.eliminate_fewer_points(raw_x, raw_y)
                    
            # sort points along y 
            in_x, in_y = sort_along_y(in_x, in_y)  

            result_image = draw_points(in_x, in_y, deepcopy(image))

            out_x.append(in_x)
            out_y.append(in_y)
            out_images.append(result_image)
            
        return out_x, out_y,  out_images

 
    def eliminate_fewer_points(self, x, y):
        # eliminate fewer points
        out_x = []
        out_y = []
        for i, j in zip(x, y):
            if len(i)>2:
                out_x.append(i)
                out_y.append(j)     
        return out_x, out_y   

    def generate_result(self, confidance, offsets,instance, thresh):

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
