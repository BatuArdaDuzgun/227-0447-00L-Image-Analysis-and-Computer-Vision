import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


class CNN(nn.Module):
    """Convolutional Neural Network.
    
    We provide a simple network with a Conv layer, followed by pooling,
    and a fully connected layer. Modify this to test different architectures,
    and hyperparameters, i.e. different number of layers, kernel size, feature
    dimensions etc.

    See https://pytorch.org/docs/stable/nn.html for a list of different layers
    in PyTorch.
    """

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(12, 18, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.do1 = nn.Dropout(0.2)  # 20% probability of an element to be zeroed
        self.fc1 = nn.Linear(18 * 6 * 6, 120)
        self.do2 = nn.Dropout(0.2)  # 20% probability of an element to be zeroed
        self.fc2 = nn.Linear(120, 40)
        self.do3 = nn.Dropout(0.1)  # 10% probability of an element to be zeroed
        self.fc3 = nn.Linear(40, 6)


    def forward(self, x):
        """Forward pass of network."""
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.do1(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        x = F.relu(self.fc2(x))
        x = self.do3(x)
        x = F.relu(self.fc3(x))
        return x


def get_transforms_train():
    """Return the transformations applied to images during training.
    
    See https://pytorch.org/vision/stable/transforms.html for a full list of 
    available transforms.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # convert image to a PyTorch Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(degrees=(-30, 30))
            # should add some transforms here to filp or rotate the image.
        ]
    )
    return transform


def get_transforms_val():
    """Return the transformations applied to images during validation.

    Note: You do not need to change this function 
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # convert image to a PyTorch Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
        ]
    )
    return transform


def get_loss_function():
    """Return the loss function to use during training. We use
       the Cross-Entropy loss for now.
    
    See https://pytorch.org/docs/stable/nn.html#loss-functions.
    """
    return nn.CrossEntropyLoss()


def get_optimizer(network, lr=0.001, momentum=0.9):
    """Return the optimizer to use during training.
    
    network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    """

    # The fist parameter here specifies the list of parameters that are
    # learnt. In our case, we want to learn all the parameters in the network
    return optim.SGD(network.parameters(), lr=lr, momentum=momentum)
