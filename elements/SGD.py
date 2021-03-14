from SGDepth.models.sgdepth import SGDepth
from SGDepth.arguments import InferenceEvaluationArguments
import torch
import torchvision.transforms as transforms
import time
import cv2
import os
from PIL import Image
import numpy as np
import glob as glob

opt = InferenceEvaluationArguments().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Inference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.num_classes = 20
        self.depth_min = opt.model_depth_min
        self.depth_max = opt.model_depth_max

        self.labels = (('CLS_ROAD', (128, 64, 128)),
                       ('CLS_SIDEWALK', (244, 35, 232)),
                       ('CLS_BUILDING', (70, 70, 70)),
                       ('CLS_WALL', (102, 102, 156)),
                       ('CLS_FENCE', (190, 153, 153)),
                       ('CLS_POLE', (153, 153, 153)),
                       ('CLS_TRLIGHT', (250, 170, 30)),
                       ('CLS_TRSIGN', (220, 220, 0)),
                       ('CLS_VEGT', (107, 142, 35)),
                       ('CLS_TERR', (152, 251, 152)),
                       ('CLS_SKY', (70, 130, 180)),
                       ('CLS_PERSON', (220, 20, 60)),
                       ('CLS_RIDER', (255, 0, 0)),
                       ('CLS_CAR', (0, 0, 142)),
                       ('CLS_TRUCK', (0, 0, 70)),
                       ('CLS_BUS', (0, 60, 100)),
                       ('CLS_TRAIN', (0, 80, 100)),
                       ('CLS_MCYCLE', (0, 0, 230)),
                       ('CLS_BCYCLE', (119, 11, 32)),
                       )
        

        with torch.no_grad():
            self.model = SGDepth()

            # load weights (copied from state manager)
            state = self.model.state_dict()
            to_load = torch.load(self.model_path)
            for (k, v) in to_load.items():
                if k not in state:
                    # print(f"    - WARNING: Model file contains unknown key {k} ({list(v.shape)})")
                    pass

            for (k, v) in state.items():
                if k not in to_load:
                    # print(f"    - WARNING: Model file does not contain key {k} ({list(v.shape)})")
                    pass

                else:
                    state[k] = to_load[k]

            self.model.load_state_dict(state)
            if torch.cuda.is_available():
                    self.model = self.model.eval().cuda()  # for inference model should be in eval mode and on gpu
            else:
                self.model = self.model.eval()
                
        print("SGD model loaded!")


    def load_image(self, frame):

        self.image = Image.fromarray(frame[...,::-1])
        self.image_o_width, self.image_o_height = self.image.size

        resize = transforms.Resize(
            (192, 640))
        image = resize(self.image)  # resize to argument size
        to_tensor = transforms.ToTensor()  # transform to tensor

        self.input_image = to_tensor(image)  # save tensor image to self.input_image for saving later
        image = self.normalize(self.input_image[:3,:,:])
        image = image.unsqueeze(0).float().to(device)

        # simulate structure of batch:
        image_dict = {('color_aug', 0, 0): image}  # dict
        image_dict[('color', 0, 0)] = image
        image_dict['purposes'] = [['segmentation', ], ['depth', ]]
        image_dict['num_classes'] = torch.tensor([self.num_classes])
        image_dict['domain_idx'] = torch.tensor(0)
        self.batch = (image_dict,)  # batch tuple


    def normalize(self, tensor):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        normalize = transforms.Normalize(mean, std)
        tensor = normalize(tensor)

        return tensor


    def inference(self, frame):

        # load image and transform it in necessary batch format
        self.load_image(frame)

        with torch.no_grad():
            output = self.model(self.batch) # forward pictures

        disps_pred = output[0]["disp", 0] # depth results
        segs_pred = output[0]['segmentation_logits', 0] # seg results

        segs_pred = segs_pred.exp().cpu()
        segs_pred = segs_pred.numpy()  # transform preds to np array
        segs_pred = segs_pred.argmax(1)  # get the highest score for classes per pixel

        depth_pred, seg_img = self.final_output(segs_pred, disps_pred) # saves results

        return depth_pred, seg_img


    def final_output(self, segs_pred, depth_pred):
        segs_pred = segs_pred[0]

        # init of seg image
        seg_img_array = np.zeros((3, segs_pred.shape[0], segs_pred.shape[1]))

        # create a color image from the classes for every pixel todo: probably a lot faster if vectorized with numpy
        # i = 0
        # while i < segs_pred.shape[0]:  # for row
        #     n = 0
        #     while n < segs_pred.shape[1]:  # for column
        #         lab = 1
        #         while lab < self.num_classes:  # for classes
        #             if segs_pred[i, n] == lab:
        #                 # write colors to pixel
        #                 seg_img_array[0, i, n] = self.labels[lab][1][0]
        #                 seg_img_array[1, i, n] = self.labels[lab][1][1]
        #                 seg_img_array[2, i, n] = self.labels[lab][1][2]
        #                 break
        #             lab += 1
        #         n += 1
        #     i += 1

        #  create a color image for Side Walk class (244, 35, 232)
        lab = 1
        seg_img_array[0, np.where(segs_pred==lab)[0], np.where(segs_pred==lab)[1]] = self.labels[lab][1][0]
        seg_img_array[1, np.where(segs_pred==lab)[0], np.where(segs_pred==lab)[1]] = self.labels[lab][1][1]
        seg_img_array[2, np.where(segs_pred==lab)[0], np.where(segs_pred==lab)[1]] = self.labels[lab][1][2]

        # scale the color values to 0-1 for proper visualization of OpenCV
        seg_img = seg_img_array.transpose(1, 2, 0).astype(np.uint8)
        seg_img = seg_img[:, :, ::-1 ]

        depth_pred = np.array(depth_pred[0][0].cpu())  # depth predictions to numpy and CPU
        depth_pred = self.scale_depth(depth_pred)  # Depthmap in meters
        depth_pred = depth_pred * (255 / depth_pred.max())  # Normalize Depth to 255 = max depth
        depth_pred = np.clip(depth_pred, 0, 255)  # Clip to 255 for safety
        
        return depth_pred, seg_img


    def scale_depth(self, disp):
        min_disp = 1 / self.depth_max
        max_disp = 1 / self.depth_min
        return min_disp + (max_disp - min_disp) * disp
