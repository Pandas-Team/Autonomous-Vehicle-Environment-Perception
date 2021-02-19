import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from CurveLanes.hourglass_network import lane_detection_network
from torch.autograd import Function as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.lane_detection_network = lane_detection_network()

    #####################################################
    ## predict lanes
    #####################################################
    def predict_lanes(self, inputs):
        inputs = torch.from_numpy(inputs).float() 
        inputs = Variable(inputs).to(device)
        
        outputs, features = self.lane_detection_network(inputs)

        return outputs


    #####################################################
    ## evaluate(test mode)
    #####################################################                                                
    def evaluate_mode(self):
        self.lane_detection_network.eval()

    #####################################################
    ## Setup GPU computation
    #####################################################                                                
    def cuda(self):
        #GPU_NUM = 1
        #device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        #torch.cuda.set_device(device) 
        self.lane_detection_network.cuda()

    #####################################################
    ## Load save file
    #####################################################
    def load_weights(self, add):
        self.lane_detection_network.load_state_dict(
            torch.load(add, map_location=device),False
        )
