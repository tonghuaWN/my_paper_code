from util import utils

import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import copy
import matplotlib.pyplot as plt
import numpy as np

# root = "/home/wn/下载/code/classfication_with_torch-master"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5,), std=(0.5,))
# ])
#
# dataset = {
#     "train": datasets.MNIST(
#         root=root,
#         transform=transform,
#         train=True,
#         download=True
#     ),
#     "test": datasets.MNIST(
#         root=root,
#         transform=transform,
#         train=False
#     )
# }
#
# dataset_size = {x: len(dataset[x]) for x in ["train", "test"]}
#
# data_loader = {
#     x: DataLoader(
#         dataset=dataset[x], batch_size=256, shuffle=True
#     ) for x in ["train", "test"]
# }


class Net(nn.Module):

    def __init__(self, in_channel):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 40, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(40),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(40, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )
        self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        # x = torch.flatten(x, start_dim=1)
        # x = self.classifier(x)
        x = self.features2(x)
        # x = torch.flatten(x, start_dim=1)
        # x = self.classifier2(x)
        x = x.reshape(x.shape[0], x.shape[1])
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)


# writer = SummaryWriter(log_dir="./log")


def relative_complexity(train_queue, num_x_bits, in_channel):
    if in_channel == 1:
        checkpoint_path = "../weight/mnist/last.pth"
    elif in_channel == 3:
        checkpoint_path = "../weight/cifar10/last.pth"
    else:
        checkpoint_path = ""
    net = Net(in_channel).cuda()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint)
    index = 0
    for step, x in enumerate(train_queue):
        x, label = utils.common_x_operations_paper(x, num_x_bits)
        inputs = x.to(device)
        labels = label.to(device)
        features = net(inputs)
        tau_0 = utils.generate_tau_0_(features, labels)
        return tau_0
        # writer.add_histogram('tau_0', tau_0, index)
        # index = index + 1
