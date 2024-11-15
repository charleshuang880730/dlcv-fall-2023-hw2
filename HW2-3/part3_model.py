# Import necessary packages.
import torch
import torch.nn as nn
import torchvision.models as models
    
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(256, 256),  # 512 is the output size from resnet18's penultimate layer
            nn.ReLU(),
            nn.Linear(256, 2)     # 2 outputs for source and target domains
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layer(x)

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        # self.resnet18 = models.resnet34(pretrained=False)
        # Remove the last FC layer for customization
        # self.backbone = nn.Sequential(*list(self.resnet18.children())[:-1])
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 256, kernel_size=3),
        )
        self.grl = GradientReversalLayer(alpha=1.0)

        self.classifier = nn.Sequential(
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 10)  # 10 classes for digits
        )
        self.domain_classifier = DomainClassifier()

    def forward(self, x, adaptation=False):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        if adaptation:
            reverse_features = self.grl(features)
            domain_output = self.domain_classifier(reverse_features)
            return domain_output
        else:
            class_output = self.classifier(features)
            return class_output
