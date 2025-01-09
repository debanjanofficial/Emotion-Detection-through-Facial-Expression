import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class EmotionNetResNet50(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, dropout_rate=0.3):
        super(EmotionNetResNet50, self).__init__()
        
        #Load the ResNet-50 model
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Modify first conv layer for grayscale input
        # Sve the Original weights
        original_weight = self.resnet.conv1.weight.data
        mean_weight = original_weight.mean(dim=1, keepdim=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Set the new conv layer weights to the mean of the original weights
        self.resnet.conv1.weight.data = mean_weight
        
        # Remove the final fully connected layer
        modules = list(self.resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
