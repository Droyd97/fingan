import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torch.autograd import grad as torch_grad
from torch.autograd import Variable
import torch.optim as optim
from torchvision import models
from pytorch_model_summary import summary
from tqdm import tqdm
import numpy as np


class RaGAN():
    """ RaGAN """
    def __init__(self, workers=2, batchSize=128, num_epochs=100, n_critic=5, lr=0.0002, beta1=0.5, ngpu=0):
        # Number of workers for dataloader
        self.workers = workers

        # Batch size during training
        self.batch = batchSize

        # Number of training epochs
        self.num_epochs = num_epochs

        # Number of critic iterations per iteration of the generator
        self.n_critic = n_critic

        # Learning rate for optimizer
        self.lr = lr

        # Beta1 hyperparameter for Adam optimizers
        self.beta1 = 0.5

        # Number of GPUs available. Use 0 for CPU mode
        self.ngpu = ngpu

        # Set device to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        # Create Generator
        self.net_g = self.Generator(self.ngpu).to(self.device)

        # Create Critic
        self.net_c = self.Critic(self.ngpu).to(self.device)

        # Noise length
        self.noiseLength = 100

        # Track Losses
        self.losses = {'g': [], 'c': [], 'gp': [], 'gradNorm': [], 'iter': 0}

        # Gradient penalty weight
        self.gp_weight = 1

        # optimisers
        self.optimiserC = optim.Adam(self.net_c.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimiserG = optim.Adam(self.net_g.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super().__init__()
            self.ngpu = ngpu
            self.input1 = AddDimension()
            self.input2 = AddDimension()
            self.gen_architecture = nn.Sequential(
                AddDimension(),
                nn.ZeroPad2d((2, 2, 0, 1)),
                spectral_norm(nn.Conv2d(1, 512, (2, 5)), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ZeroPad2d((2, 2, 0, 1)),
                spectral_norm(nn.Conv2d(512, 256, (2, 5)), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ZeroPad2d((2, 2, 0, 1)),
                spectral_norm(nn.Conv2d(256, 128, (2, 5)), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ZeroPad2d((2, 2, 0, 1)),
                spectral_norm(nn.Conv2d(128, 64, (2, 5)), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ZeroPad2d((2, 2, 0, 1)),
                spectral_norm(nn.Conv2d(64, 32, (2, 5)), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ZeroPad2d((2, 2, 0, 1)),
                spectral_norm(nn.Conv2d(32, 16, (2, 5)), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ZeroPad2d((2, 2, 0, 1)),
                spectral_norm(nn.Conv2d(16, 8, (2, 5)), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv2d(8, 4, (3, 3), padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv2d(4, 2, (3, 3), padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv2d(2, 1, (2, 1), padding=0), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                SqueezeDimension(),
                SqueezeDimension()
            )

        def forward(self, input1, input2):
            combine = torch.stack((input1, input2), dim=1)
            return self.gen_architecture(combine)

    class Critic(nn.Module):
        def __init__(self, ngpu):
            super().__init__()
            self.ngpu = ngpu
            self.critic_architecture = nn.Sequential(
                AddDimension(),

                spectral_norm(nn.Conv1d(1, 64, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(64, 64, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(64, 64, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(64, 64, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(64, 32, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                spectral_norm(nn.Conv1d(32, 1, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                SqueezeDimension(),
                nn.Linear(200, 100),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(100, 50),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(50, 1)
            )

        def forward(self, input1, input2):
            combine = torch.cat((input1, input2), dim=1)
            # print(combine.shape)
            return self.critic_architecture(combine)

    def train(self, data, num_epochs):
        dataloader = DataLoader(data, 10)

        print("Starting Training Loop...")

        for epoch in tqdm(range(num_epochs)):
            for i, data in enumerate(dataloader, 0):
                loss = nn.BCEWithLogitsLoss()

                base1 = data[:5, :, 0]
                base2 = data[5:, :, 0]
                associated1 = data[:5, :, 1]
                associated2 = data[5:, :, 1]
                # print(base1.shape, base2.shape, associated.shape)
                b_size = base1.size()[0]

                # target.data.resize_(base.shape.detach()).fill_(1)

                for _ in range(self.n_critic):
                    noise = torch.randn(b_size, self.noiseLength, device=self.device)
                    generated_data = self.net_g(Variable(base1.float()), noise)

                    self.optimiserC.zero_grad()

                    c_real_av = self.net_c(Variable(base1.float()), associated1.float()).mean()
                    c_fake_av = self.net_c(generated_data, associated1.float()).mean()

                    c_real = self.net_c(Variable(base1.float()), Variable(associated1.float()))
                    c_fake = self.net_c(generated_data, Variable(associated1.float()))
                    # print(c_real_av.shape, c_fake_av.shape, c_real.shape, c_fake.shape, base.shape)
                    # print(summary(self.Critic(0), base.float(), associated.float()))
                    target_real = torch.FloatTensor(c_real.shape).fill_(1)
                    target_fake = torch.FloatTensor(c_fake.shape).fill_(0)
                    # x = self.net_g(base.float(), noise)
                    # print(self.net_c(self.net_g(base.float(), noise), associated.float()).mean())
                    # print(summary(self.Generator(0), base.float(), noise, show_input=False))

                    d_loss = (loss(c_real - c_fake_av, target_real)
                              + loss(c_fake - c_real_av, target_fake)) / 2
                    d_loss.backward()

                    self.optimiserC.step()

                self.losses['c'].append(d_loss.detach().numpy())

                b_size = base2.size()[0]
                noise = torch.randn(b_size, self.noiseLength, device=self.device)
                generated_data = self.net_g(Variable(base2.float()), noise)

                self.optimiserG.zero_grad()

                c_real_av = self.net_c(Variable(base2.float()), Variable(associated2.float())).mean()
                c_fake_av = self.net_c(generated_data, associated2.float()).mean()

                c_real = self.net_c(Variable(base2.float()), Variable(associated2.float()))
                c_fake = self.net_c(generated_data, Variable(associated2.float()))

                target_real = torch.FloatTensor(c_real.shape).fill_(1)
                target_fake = torch.FloatTensor(c_fake.shape).fill_(0)

                g_loss = (loss(c_real - c_fake_av, target_fake)
                          + loss(c_fake - c_real_av, target_real)) / 2
                    
                g_loss.backward()
                self.losses['g'].append(g_loss.detach().numpy())
                self.optimiserG.step()

    def generate(self, base):
        noise = torch.randn(base.shape[0], self.noiseLength, device=self.device)
        print(type(base), type(noise))
        return self.net_g(Variable(base.float()), noise)


class AddDimension(nn.Module):
    """ AddDimension """
    def forward(self, x):
        return x.unsqueeze(1)


class SqueezeDimension(nn.Module):
    """ Squeeze Dimension """
    def forward(self, x):
        return x.squeeze(1)
