import torch
from torchvision import models



class ModelLoader_types(object):
    model = None

    def __init__(self,num_classes, ckpt_path=None, model_type="segmentation"):

        if model_type == "classification":
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.num_classes = num_classes
            self.model.fc = torch.nn.Linear(num_ftrs, self.num_classes)
        else:
            self.model = models.segmentation.fcn_resnet101(pretrained=True)
            self.model.classifier = torch.nn.Conv2d(
                in_channels=2048,
                out_channels=1,
                kernel_size=1,
                stride=1)

        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])


    def load_model(self, ckpt_path,model_type):

        if model_type == "classification":
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, self.num_classes)
        else:
            self.model = models.segmentation.fcn_resnet101(pretrained=True)
            self.model.classifier = torch.nn.Conv2d(
                in_channels=2048,
                out_channels=1,
                kernel_size=1,
                stride=1)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def run_inference(self, input_data):
        if self.model is None:
            raise Exception("no valid model was loaded, can't perform inference")
        with torch.no_grad():
            output = self.model.forward(input_data)
        return output

    def to_device(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()



class ModelLoader_simple(object):
    model = None

    def __init__(self, ckpt_path=None):
        self.model = torch.load(ckpt_path,map_location=torch.device('cpu'))

    def load_model(self, ckpt_path):
        self.model = torch.load(ckpt_path,map_location=torch.device('cpu'))

    def run_inference(self, input_data):
        if self.model is None:
            raise Exception("no valid model was loaded, can't perform inference")
        with torch.no_grad():
            output = self.model.forward(input_data)
        return output

    def to_device(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()