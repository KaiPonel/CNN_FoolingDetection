import torch.nn as nn
    
class CifarCNN(nn.Module):
    """
        Defines a CNN for the CIFAR10 dataset. \n
        The network consists of 3 convolutional layers, each followed by a ReLU activation function. \n
        After each convolutional layer, a max pool layer is applied. \n
        The final layer is a fully connected layer with 10 outputs, one for each class.
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.network(x)
        
class MnistCNN(nn.Module):
    """
        Defines a CNN for the MNIST dataset. \n
        The network consists of 3 convolutional layers, each followed by a ReLU activation function. \n
        After each convolutional layer, a max pool layer is applied. \n
        The final layer is a fully connected layer with 10 outputs, one for each class.
     """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Flatten(),
        nn.Linear(256 * 3 * 3, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    def forward(self, x):
        return self.network(x)



class MLP(nn.Module):
    """
        Defines a MLP for the MNIST dataset. \n
        The network consists of 3 fully connected layers, each followed by a ReLU activation function. \n
        The final layer is a fully connected layer with 10 outputs, one for each class.
    """
    def __init__(self, img_dim_x, img_dim_y, img_dim_z, amount_classes):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_dim_x * img_dim_y * img_dim_z, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, amount_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

def get_model(model_type="cnn", dataset="mnist"):
    """
    Returns the (untrained) model for the given model type and dataset.
    `model_type`: type of model. allowed values are: ["cnn", "mlp", "imagenet"] \n
    `dataset`: dataset the model will be trained on. allowed values are: ["mnist", "cifar10", "imagenet"] \n
    """
    if model_type not in ["cnn", "mlp", "imagenet"]:
        raise Exception("Invalid value for `model_type`. Got {}, expected [`cnn`, `mlp`, `imagenet`]".format(model_type))
    if dataset not in ["mnist", "cifar10", "imagenet"]:
        raise Exception("Invalid value for `dataset`. Got {}, expected [`mnist`, `cifar10`, `imagenet`]".format(dataset))

    if model_type == "cnn":
        if dataset=="mnist":
            return MnistCNN()
        if dataset=="cifar10":
            return CifarCNN()
        
    if model_type == "imagenet":
        if dataset=="mnist":
            return ResNet50()
        if dataset=="cifar10":
            return ResNet50()
        
    if model_type == "mlp":
        if dataset=="mnist":
            return MLP(28, 28, 1, 10)
        if dataset=="cifar10":
            return MLP(32, 32, 3, 10)
        
def get_modelCifarCNN(model_type="cnn", dataset="cifar10"):
    """
    Returns the (untrained) model for the given model type and dataset.
    `model_type`: type of model. allowed values are: ["cnn", "mlp", "imagenet"] \n
    `dataset`: dataset the model will be trained on. allowed values are: ["mnist", "cifar10", "imagenet"] \n
    """
    if model_type not in ["cnn", "mlp", "imagenet"]:
        raise Exception("Invalid value for `model_type`. Got {}, expected [`cnn`, `mlp`, `imagenet`]".format(model_type))
    if dataset not in ["mnist", "cifar10", "imagenet"]:
        raise Exception("Invalid value for `dataset`. Got {}, expected [`mnist`, `cifar10`, `imagenet`]".format(dataset))

    return CifarCNN()
        
