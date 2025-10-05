from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

# Transformation (same as in notebook)
custom_transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, features, labels, transform):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # resize to (28,28)
        image = self.features[index].reshape(28,28)
        # change datatype (np.uint8)
        image = image.astype(np.uint8)
        # convert 1D grayscale to 3-channel RGB
        image = np.stack([image]*3, axis=-1)
        # convert array to PIL image
        image = Image.fromarray(image)
        # apply transforms
        image = self.transform(image)
        return image, torch.tensor(self.labels[index], dtype=torch.long)
