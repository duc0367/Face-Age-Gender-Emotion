import torch
import torch.nn as nn
import torch.nn.functional as F


class AgeGenderNet(nn.Module):
    def __init__(self):
        super(AgeGenderNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding='same')  # (B, 32, 200, 200)
        self.bn1 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)  # (B, 32, 100, 100)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')  # (B, 64, 100, 100)
        self.bn2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)  # (B, 64, 50, 50)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same')  # (B, 128, 50, 50)
        self.bn3 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)  # (B, 128, 25, 25)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same')  # (B, 256, 25, 25)
        self.bn4 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, padding=1)  # (B, 256, 13, 13)

        self.flatten = nn.Flatten()

        self.linear11 = nn.Linear(43264, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.linear12 = nn.Linear(256, 1)

        self.linear21 = nn.Linear(43264, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.linear22 = nn.Linear(256, 1)

    def forward(self, x):
        assert x.shape[1:] == (3, 200, 200)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        assert x.shape[1:] == (32, 100, 100)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        assert x.shape[1:] == (64, 50, 50)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        assert x.shape[1:] == (128, 25, 25)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        print(x.shape)
        assert x.shape[1:] == (256, 13, 13)

        x = self.flatten(x)
        print(x.shape)

        out1 = self.linear11(x)
        out1 = self.dropout1(out1)
        out1 = self.linear12(out1)
        out1 = F.sigmoid(out1)  # gender

        out2 = self.linear21(x)
        out2 = self.dropout2(out2)
        out2 = self.linear22(out2)
        out2 = F.relu(out2)  # age

        return out1, out2


if __name__ == "__main__":
    test = torch.randn(1, 3, 200, 200)
    model = AgeGenderNet()
    print(model(test))
