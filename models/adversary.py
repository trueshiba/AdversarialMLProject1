import torch
import torch.nn as nn

class AttackModel(nn.Module):

    def __init__(self, num_classes=2):
        super(AttackModel, self).__init__()

        # self.features = nn.Sequential(
        #     nn.Linear(700,1024),
        #     #nn.MaxPool2d(2),
        #     nn.ReLU(),
        #     nn.Linear(1024,512),
        #     nn.ReLU(),
        #     nn.Linear(512,256),
        #     nn.ReLU(),
        #     nn.Linear(256,128),
        #     nn.Sigmoid(),
        # )
        # self.classifier = nn.Linear(128,num_classes)

        self.features = nn.Sequential(
            nn.Linear(200, 1024),
            nn.ReLU(),
            #nn.BatchNorm1d(1024), 
            nn.Dropout(0.3),  

            nn.Linear(1024, 512),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128,num_classes)

    def forward(self,x):
        torch.device('cuda:0')

        hidden_out = self.features(x)
        out = self.classifier(hidden_out)
        return torch.sigmoid(out)
 


 