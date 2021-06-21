import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torch.autograd import grad as torch_grad
from torch.autograd import Variable
import torch.optim as optim
from torchvision import models
from pytorch_model_summary import summary
from tqdm import tqdm


# Taken from....
class ResNet():
    """ ResNet """
    def __init__(self, lr=0.0002, beta1=0.5, ngpu=0):

        # Learning rate for optimizer
        self.lr = lr

        # Beta1 hyperparameter for Adam optimizers
        self.beta1 = 0.5

        # Number of GPUs available. Use 0 for CPU mode
        self.ngpu = ngpu

        # Set device to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        # Track Losses
        self.losses = {'train': [], 'test': []}

        self.model = self.Model(self.ngpu).to(self.device)

        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

    class Model(nn.Module):
        def __init__(self, ngpu):
            super().__init__()
            self.ngpu = ngpu

            # Block 1
            self.block1 = nn.Sequential(
                AddDimension(),
                nn.ConstantPad1d((4, 3), 0),
                spectral_norm(nn.Conv1d(1, 64, 8, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(64, 64, 5, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(64, 64, 3, padding=1)),
            )
            self.shortcut1 = nn.Sequential(
                AddDimension(),
                spectral_norm(nn.Conv1d(1, 64, 1, padding=0)))

            # Block 2
            self.block2 = nn.Sequential(
                nn.ConstantPad1d((4, 3), 0),
                spectral_norm(nn.Conv1d(64, 128, 8, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(128, 128, 5, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(128, 128, 3, padding=1)),
            )
            self.shortcut2 = spectral_norm(nn.Conv1d(64, 128, 1, padding=0))

            # Block 3
            self.block3 = nn.Sequential(
                nn.ConstantPad1d((4, 3), 0),
                spectral_norm(nn.Conv1d(128, 128, 8, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(128, 128, 5, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(128, 128, 3, padding=1)),
            )
            # self.shortcut3 = spectral_norm()

            self.output = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                RemoveDimension(),
                nn.Linear(128, 2),
                nn.Softmax(dim=1),
            )

        def forward(self, x):
            # Block 1
            input = x
            # print(input.shape)
            x = self.block1(input)
            y = self.shortcut1(input)
            x = F.relu(torch.add(x, y))
            output = x

            # Block 2
            # print(output.shape)
            x = self.block2(output)
            # print(x.shape)
            y = self.shortcut2(output)
            # print(y.shape)
            x = F.relu(torch.add(x, y))
            output = x

            # Block 3
            x = self.block3(output)
            # y = self.shortcut3(output)
            x = F.relu(torch.add(x, output))

            # Final
            return self.output(x)

    def train(self, train_data, test_data, num_epochs):

        load_train = DataLoader(train_data, 10)
        # load_val = DataLoader(val_data, 10)

        print("Starting Training Loop...")

        loss = nn.CrossEntropyLoss()

        for epoch in tqdm(range(num_epochs)):

            train_loss = 0
            for i, data in enumerate(load_train, 0):
                b_size = data[:][0].size()[0]
                self.optimiser.zero_grad()
                # print('output', self.model(data[0]).shape)
                # print(data[0].shape)
                # print(summary(self.Model(0), data[0].float()))
                # print(self.model(data[0]).shape)
                # print(data[1].shape)
                output = loss(self.model(data[0]), data[1].squeeze(1).type(torch.LongTensor))

                train_loss = (train_loss + output.data.item()) / 2

                output.backward()

                self.optimiser.step()

            self.losses['train'].append(train_loss)

            # val loss

            output = loss(self.model(test_data[:][0]), test_data[:][1].squeeze(1).type(torch.LongTensor))

            self.losses['test'].append(output.data.item())

    def predict(self, x_val):
        return self.model(x_val)


class AddDimension(nn.Module):
    """ AddDimension """
    def forward(self, x):
        return x.unsqueeze(1)


class SqueezeDimension(nn.Module):
    """ Squeeze Dimension """
    def forward(self, x):
        return x.squeeze(1)


class RemoveDimension(nn.Module):
    def forward(self, x):
        return x.squeeze(-1)


class PrintShape(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x

