import torch
import torchvision.models as models

# state_dict is a dictionary that stores the learned parameters in a internal state.
model = models.vgg16(weights='IMAGENET1K_V1')  
#  VGG-16 is a popular convolutional neural network architecture often used for image classification tasks
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval() #  model.eval() method is used to set the model to evaluation mode.

torch.save(model, 'model.pth')

model = torch.load('model.pth')