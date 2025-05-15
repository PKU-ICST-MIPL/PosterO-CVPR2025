import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class design_intent_detector(nn.Module):
    def __init__(self, act='Sigmoid', action='forward'):
        super(design_intent_detector, self).__init__()
        self.model = smp.Unet(
            encoder_name="mit_b1",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        if act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'none':
            self.act = nn.Identity()
        else:
            raise NotImplementedError(act)
        
        if action == 'forward':
            self.forward = self.feed_forward
        elif action == 'extract':
            self.forward = self.encode_feature
        else:
            raise NotImplementedError(action)
    
    def encode_feature(self, x):
        return self.model.encoder(x)[-1] # B, 512, 7, 7

    def feed_forward(self, x):
        output = self.model(x)
        output = self.act(output)
        return output