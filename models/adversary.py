import torch
import torch.nn as nn

class AttackModel(nn.Module):

    def __init__(self, num_classes=2):
        super(AttackModel, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(700,1024),
            #nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(128,num_classes)

    def forward(self,x):
        torch.device('cuda:0')

        hidden_out = self.features(x)
        out = self.classifier(hidden_out)
        return out