import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functools import reduce
from operator import __add__

KERNEL_SIZE = (4, 4)


# This is used to implement a `same` padding like we'd have in Tensorflow
# For some reasons, padding dimensions are reversed wrt kernel sizes,
# first comes width then height in the 2D case.
#
# Based on [this](https://stackoverflow.com/a/63149259/9347193) StackOverflow answer
conv_padding = reduce(__add__,
                      [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in KERNEL_SIZE[::-1]])


class FINNger(nn.Module):
    def __init__(self, num_classes, learning_rate, weight_decay):
        super(FINNger, self).__init__()

        self.pad1_1 = nn.ZeroPad2d(conv_padding)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=KERNEL_SIZE)
        self.batchnorm1_1 = nn.BatchNorm2d(64)
        self.pad1_2 = nn.ZeroPad2d(conv_padding)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=KERNEL_SIZE)
        self.batchnorm1_2 = nn.BatchNorm2d(64)
        self.maxpooling1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.2)

        self.pad2_1 = nn.ZeroPad2d(conv_padding)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=KERNEL_SIZE)
        self.batchnorm2_1 = nn.BatchNorm2d(128)
        self.pad2_2 = nn.ZeroPad2d(conv_padding)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=KERNEL_SIZE)
        self.batchnorm2_2 = nn.BatchNorm2d(128)
        self.maxpooling2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.3)

        self.pad3_1 = nn.ZeroPad2d(conv_padding)
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=KERNEL_SIZE)
        self.batchnorm3_1 = nn.BatchNorm2d(128)
        self.pad3_2 = nn.ZeroPad2d(conv_padding)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=KERNEL_SIZE)
        self.batchnorm3_2 = nn.BatchNorm2d(128)
        self.maxpooling3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(0.4)

        self.flatten = nn.Flatten()

        # Image starts with 96, and we have 3 maxpools with kernel size 2
        image_size_dense = 96 // 2 // 2 // 2

        # Image is squared, and we have 128 layers from the convolutions
        # We also output 128 layers to the output one, which then converts to num_classes
        dense_out = 128
        self.dense = nn.Linear(
            image_size_dense * image_size_dense * 128, dense_out)
        self.out = nn.Linear(dense_out, num_classes)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def forward(self, x):
        # Sequence of convolutions with ReLU activations
        # Shape starts with (BATCH, 3, 96, 96)
        # out: BATCH, 64, 96, 96
        x = self.batchnorm1_1(F.relu(self.conv1_1(self.pad1_1(x))))
        # out: BATCH, 64, 96, 96
        x = self.batchnorm1_2(F.relu(self.conv1_2(self.pad1_2(x))))
        x = self.maxpooling1(x)  # out: BATCH, 64, 48, 48
        x = self.dropout1(x)  # out: BATCH, 64, 48, 48

        # out: BATCH, 128, 48, 48
        x = self.batchnorm2_1(F.relu(self.conv2_1(self.pad2_1(x))))
        # out: BATCH, 128, 48, 48
        x = self.batchnorm2_2(F.relu(self.conv2_2(self.pad2_2(x))))
        x = self.maxpooling2(x)  # out: BATCH, 128, 24, 24
        x = self.dropout2(x)  # out: BATCH, 128, 24, 24

        # out: BATCH, 128, 24, 24
        x = self.batchnorm3_1(F.relu(self.conv3_1(self.pad3_1(x))))
        # out: BATCH, 128, 24, 24
        x = self.batchnorm3_2(F.relu(self.conv3_2(self.pad3_2(x))))
        x = self.maxpooling3(x)  # out: BATCH, 128, 12, 12
        x = self.dropout3(x)  # out: BATCH, 128, 12, 12

        x = self.flatten(x)  # out: BATCH, 18432
        x = F.relu(self.dense(x))  # out: BATCH, 128
        x = self.out(x)  # out: BATCH, NUM_CLASSES

        return F.log_softmax(x, dim=1)

    def save(self, model_id: str = ''):
        torch.save(self.state_dict(), f'model/model{model_id}.pth')
        torch.save(self.optimizer.state_dict(),
                   f'model/optimizer{model_id}.pth')

    def load(self, model_id: str = ''):
        try:
            print("Loading from ", f'model/model{model_id}.pth')
            self.load_state_dict(torch.load(f'model/model{model_id}.pth'))
            self.optimizer.load_state_dict(
                torch.load(f'model/optimizer{model_id}.pth'))
            self.eval()
            print("Model loaded successfuly")
        except FileNotFoundError as error:
            error.strerror = "There is no model located on"
            raise error from None

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
