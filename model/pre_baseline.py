import torch
import torch.nn as nn
from model.convnet4 import Conv4
from model.resnet12 import resnet12
from model.WRN28 import WideResNet
from model.classifier import LinearClassifier,CosClassifier
import utils

class PreBaseline(nn.Module):   
    def __init__(self, config):
        super().__init__()
        if config['model_args']['encoder'] == 'wrn':    
            self.encoder = WideResNet()
        elif config['model_args']['encoder'] == 'resnet12': 
            self.encoder = resnet12()
        else config['model_args']['encoder'] == 'conv4'
            self.encoder = Conv4()

        in_dim = self.encoder.out_dim
        
        if config['model_args']['classifier'] == 'Linear':
            utils.log("use Linear Classification")
            self.classifier = LinearClassifier(in_dim=in_dim, n_classes=config['model_args']['classifier_n_classes'])
        elif config['model_args']['classifier'] == 'Cos':
            utils.log("use Cos Classification")
            self.classifier = CosClassifier(in_dim=in_dim, n_classes=config['model_args']['classifier_n_classes'], metric='cos', temper=None)
        else:
            utils.log("No Classifier")

        self.rot_classifier = LinearClassifier(in_dim=in_dim, n_classes=4)

    def forward(self, x):
        x = self.encoder(x)        
        logits_class = self.classifier(x)
        logits_rot = self.rot_classifier(x)
        return logits_class, logits_rot
