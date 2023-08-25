import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

'''
Transforms are common image transformations available in the torchvision.transforms module. 
They can be chained together using Compose. Most transform classes have a function equivalent:
functional transforms give fine-grained control over the transformations. 
This is useful if you have to build a more complex transformation pipeline
'''

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

