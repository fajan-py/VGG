import torch
import torch.nn as nn


class Block_double(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.pool(self.relu(self.conv1_2(self.relu(self.conv1_1(x)))))
        return x
    

class Block_quad(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=3, padding=1 )
        self.conv3_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = (self.relu(self.conv3_1(x)))
        x = (self.relu(self.conv3_2(x)))
        x = (self.relu(self.conv3_3(x)))
        x = (self.relu(self.conv3_4(x)))
        return self.pool(x)


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 11),
        )
        self.block1 = Block_double(in_channels=1, out_channels=64)
        self.block2 = Block_double(in_channels=64, out_channels=128)
        self.block3 = Block_quad(in_channels=128, out_channels=256)
        self.block4 = Block_quad(in_channels=256, out_channels=512)
        self.block5 = Block_quad(in_channels=512, out_channels=512)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
