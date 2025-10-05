import torch.nn as nn
import torchvision.models as models

# Load pretrained VGG16
vgg16 = models.vgg16(pretrained=True)

# Freeze feature extractor layers
for param in vgg16.features.parameters():
    param.requires_grad = False

# Replace classifier layers
vgg16.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=1024, out_features=512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=512, out_features=10),
)
