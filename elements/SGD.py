from SGDepth.models.sgdepth import SGDepth
from SGDepth.arguments import InferenceEvaluationArguments
import torch
import torchvision.transforms as transforms

opt = InferenceEvaluationArguments().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SGDepth_Model:
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
        
        self.STEREO_SCALE_FACTOR = 6

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
                    self.model = self.model.eval().cuda() 
            else:
                self.model = self.model.eval()
                
        print("SGDepth model loaded!")


    def load_image(self, frame):
            
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = torch.nn.Sequential(
            transforms.Resize((192, 640)),
            transforms.Normalize(mean, std))

        frame = transforms.ToTensor()(frame[...,::-1].copy())
        image = transform(frame)
        image = image.unsqueeze(0).to(device)

        # structure of batch:
        image_dict = {('color_aug', 0, 0): image}  
        image_dict[('color', 0, 0)] = image
        image_dict['purposes'] = [['segmentation', ], ['depth', ]]
        image_dict['num_classes'] = torch.tensor([self.num_classes])
        image_dict['domain_idx'] = torch.tensor(0)
        self.batch = (image_dict,)


    def inference(self, frame):

        self.load_image(frame)
        
        with torch.no_grad():
            output = self.model(self.batch)

        disps_pred = output[0]["disp", 0] # depth results
        segs_pred = output[0]['segmentation_logits', 0] # seg results
        
        seg_img = torch.argmax(segs_pred.exp(), dim=1)
        depth_pred = self.final_output(disps_pred)

        return depth_pred.cpu().numpy(), seg_img[0].cpu().numpy()


    def final_output(self, depth_pred):
        depth_pred = self.scale_depth(depth_pred[0][0])  # Depthmap in meters
        depth_pred = self.STEREO_SCALE_FACTOR / depth_pred
        depth_pred = torch.clip(depth_pred, 0, self.depth_max)
        return depth_pred


    def scale_depth(self, disp):
        min_disp = 1 / self.depth_max
        max_disp = 1 / self.depth_min
        return min_disp + (max_disp - min_disp) * disp
