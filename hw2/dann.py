import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

d = 512

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        conv_dim = 64

        self.conv1 = nn.Conv2d(3, conv_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv4 = nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(conv_dim * 2, conv_dim * 2, kernel_size=3, padding=1)
        self.pool6 = nn.MaxPool2d(2, stride=2)

        self.conv7 = nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(conv_dim * 4, conv_dim * 4, kernel_size=3, padding=1)
        self.pool9 = nn.MaxPool2d(2, stride=2)

        self.flat_dim = conv_dim * 36
        self.fc1 = nn.Linear(self.flat_dim, d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool6(x)

        x = F.relu(self.conv7(x))
        
        x = F.relu(self.conv8(x))
        x = self.pool9(x)
        x = x.view(-1, self.flat_dim)
        x = F.relu(self.fc1(x))

        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(d, 10)
        # self.fc2 = nn.Linear(256, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.l1 = nn.Linear(d, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x, alpha=-1):
        x = ReverseLayerF.apply(x, alpha)
        x = F.leaky_relu(self.l1(x), 0.2)
        x = F.leaky_relu(self.l2(x), 0.2)

        x =  F.softmax(self.l3(x),dim=1)
        # x = torch.sigmoid(self.l3(x))
        return x
