import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same')  # (B, 64, 48, 48)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)  # (B, 64, 24, 24)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same')  # (B, 128, 24, 24)
        self.bn2 = nn.BatchNorm2d(128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)  # (B, 128, 12, 12)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same')  # (B, 256, 12, 12)
        self.bn3 = nn.BatchNorm2d(256)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)  # (B, 256, 6, 6)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(9216, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(256, 7)

    def forward(self, x):
        assert x.shape[1:] == (3, 48, 48)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        assert x.shape[1:] == (64, 24, 24)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        assert x.shape[1:] == (128, 12, 12)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        assert x.shape[1:] == (256, 6, 6)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.softmax(x, dim=1)

        return x

    def save(self, filename='model.pth'):
        folder_path = '../emotion-model'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        file_path = os.path.join(folder_path, filename)
        torch.save(self.state_dict(), file_path)


if __name__ == "__main__":
    test = torch.randn(1, 3, 48, 48)
    model = EmotionNet()
    print(model(test))
