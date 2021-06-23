import os
import logging
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torch.autograd import grad as torch_grad
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm

from fingan.models.model import Model


class WGAN_GP(Model):
    """ WGAN """
    def __init__(self, wkdir=None, is_logging=False, workers=2, batch_size=128,
                 num_epochs=200, save_point=50, ngpu=0, n_critic=5, lr=0.0002, beta1=0.5):

        super().__init__('WGAN_GP', wkdir, is_logging, batch_size, workers, num_epochs, save_point, ngpu)

        # Set device to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

        # Number of critic iterations per iteration of the generator
        self.n_critic = n_critic

        # Learning rate for optimizer
        self.lr = lr

        # Beta1 hyperparameter for Adam optimizers
        self.beta1 = 0.5

        # Create Generator
        self.net_g = self.Generator(self.ngpu).to(self.device)

        # Create Critic
        self.net_c = self.Critic(self.ngpu).to(self.device)

        # Noise length
        self.noiseLength = 50

        # Track Losses
        self.losses = {'g': torch.zeros(num_epochs), 'c': torch.zeros(num_epochs), 'gp': [], 'gradNorm': [], 'iter': 0}

        # Gradient penalty weight
        self.gp_weight = 0.001

        # optimisers
        self.optimiserC = optim.Adam(self.net_c.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimiserG = optim.Adam(self.net_g.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super().__init__()
            self.ngpu = ngpu
            self.genArchitecture = nn.Sequential(
                nn.Linear(50, 100),
                nn.LeakyReLU(0.2, inplace=True),
                AddDimension(),
                spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10),
                nn.Upsample(200),

                spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(400),

                spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(800),

                spectral_norm(nn.Conv1d(32, 1, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),

                SqueezeDimension(),
                nn.Linear(800, 100)
            )

        def forward(self, input):
            return self.genArchitecture(input)

    class Critic(nn.Module):
        def __init__(self, ngpu):
            super().__init__()
            self.ngpu = ngpu
            self.criticArchitecture = nn.Sequential(
                AddDimension(),
                spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool1d(2),

                spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool1d(2),

                spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),

                nn.Linear(800, 50),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(50, 15),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(15, 1)
            )

        def forward(self, input):
            return self.criticArchitecture(input)

    def train(self, dataset, name=None):
        # Set working directory to current one if not provided
        if self.wkdir is None:
            self.wkdir = os.getcwd()

        # Set file path
        if name is None:
            file_path = self.wkdir + self.name + "-save.pt"
        else:
            file_path = self.wkdir + name + "-save.pt"

        start_epoch = 0

        if os.path.isfile(file_path):
            epoch = self.load(file_path)
            start_epoch = epoch - 1
            print("Loaded saved model")
        
        dataloader = DataLoader(dataset, self.batch_size)

        print("Starting Training Loop...")

        for epoch in tqdm(range(start_epoch, self.num_epochs)):

            for i, data in enumerate(dataloader, 0):
                # Critic
                b_size = data.size()[0]

                cReal = self.net_c(data)

                noise = torch.randn(b_size, self.noiseLength, device=self.device)
                cGenerated = self.net_c(self.net_g(noise))

                gp = self.gradient_penalty(data, self.net_g(noise))

                # Get loss
                self.optimiserC.zero_grad()
                d_loss = (cGenerated.mean() - cReal.mean()) + gp
                d_loss.backward()
                self.optimiserC.step()

                if i % self.n_critic == 0:
                    # Generator
                    self.optimiserG.zero_grad()
                    noise = torch.randn(b_size, self.noiseLength, device=self.device)

                    gGenerated = self.net_c(self.net_g(noise))
                    g_loss = - gGenerated.mean()
                    g_loss.backward()
                    self.optimiserG.step()

            self.losses['c'][epoch] = d_loss.data.item()
            self.losses['g'][epoch] = g_loss.data.item()

            if self.is_logging:
                logging.info("Critic loss: {}".format(self.losses['c'][epoch]))
                logging.info("Generator loss: {}".format(self.losses['g'][epoch]))

            if (epoch + 1) % self.save_point == 0:
                self.save(epoch + 1, path=file_path)

    def generate(self, number):
        noise = torch.randn(number, self.noiseLength, device=self.device)
        gGenerated = self.net_g(noise)
        return gGenerated

    def save(self, epoch, path=None):
        if path is None:
            path = self.wkdir
        torch.save({
            'epoch': epoch,
            'critic_state_dict': self.net_c.state_dict(),
            'generator_state_dict': self.net_g.state_dict(),
            'critic_optimizer_state_dict': self.optimiserC.state_dict(),
            'generator_optimizer_state_dict': self.optimiserG.state_dict(),
            # 'loss': loss,
        }, path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.net_c.load_state_dict(checkpoint['critic_state_dict'])
        self.net_g.load_state_dict(checkpoint['generator_state_dict'])
        self.optimiserC.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.optimiserG.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
        # loss = checkpoint['loss']

    def summary(self):
        pass

    def wasserstein_loss(self):
        pass

    def gradient_penalty(self, real_data, generated_data):
        b_size = real_data.size()[0]

        alpha = torch.rand(b_size, 1).expand_as(real_data)

        interpolated = Variable(alpha * real_data.data + (1 - alpha) * generated_data.data, requires_grad=True)

        prob_interploated = self.net_c(interpolated)

        gradients = torch_grad(outputs=prob_interploated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interploated.size()), create_graph=True,
                               retain_graph=True)[0]

        gradients = gradients.view(b_size, -1)
        self.losses['gradNorm'].append(gradients.norm(2, dim=1).mean().data.item())

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()


class AddDimension(nn.Module):
    """ AddDimension """
    def forward(self, x):
        return x.unsqueeze(1)


class SqueezeDimension(nn.Module):
    """ Squeeze Dimension """
    def forward(self, x):
        return x.squeeze(1)
