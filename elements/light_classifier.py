import torch
import timm
import numpy as np
from torch import nn
import torch
from torchvision import transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LightModel(nn.Module):
    def __init__(self, model_name = 'regnety_002'):
        super().__init__()
        self.cnn = timm.create_model(model_name, pretrained = False, num_classes = 4)

    def forward(self, x):
        x = self.cnn(x)
        return x


class light_classifier():
    def __init__(self, model_path, model_name = 'regnety_002'):

        self.light_model = LightModel(model_name).to(device)
        self.light_model.load_state_dict(torch.load(model_path, map_location=device))
        self.light_model = self.light_model.eval()
        
        print("Traffic light classifier loaded!")

        self.light_classes = {0: 'yellow', 1: 'off', 2: 'green', 3: 'red'}
        self.transform = transforms.Compose([
                                transforms.Resize((56,28)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225])])
                
    def classify(self, img):

        with torch.no_grad():
            img = Image.fromarray(img[...,::-1])
            img = self.transform(img)
            X = img[None,...].to(device)
            pred = self.light_model(X.float())       

            y_pred = self.softmax(pred.detach().cpu().numpy()) 
            y_pred = np.argmax(y_pred, axis = 1) 
                
        return self.light_classes[y_pred[0]] 

    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1)[:, None]

        