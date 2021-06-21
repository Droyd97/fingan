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
        self.netG = self.Generator(self.ngpu).to(self.device)

        # Create Critic
        self.netC = self.Critic(self.ngpu).to(self.device)

        # Noise length
        self.noiseLength = 50

        # Track Losses
        self.losses = {'g': [], 'c': [], 'gp': [], 'gradNorm': [], 'iter': 0}

        # Gradient penalty weight
        self.gp_weight = 0.001

        # optimisers
        self.optimiserC = optim.Adam(self.netC.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimiserG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

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
            # x = input
            # for layer in self.criticArchitecture:
            #     x = layer(x)
            #     print(x.size())
            return self.criticArchitecture(input)

    def train(self, dataset, numEpochs):
        dataloader = DataLoader(dataset, 10)

        print("Starting Training Loop...")

        for epoch in tqdm(range(numEpochs)):

            for i, data in enumerate(dataloader, 0):
                # Critic
                # self.netC.zero_grad()

                # Format batch

                b_size = data.size()[0]

                cReal = self.netC(data)

                noise = torch.randn(b_size, self.noiseLength, device=self.device)
                cGenerated = self.netC(self.netG(noise))

                gp = self.gradient_penalty(data, self.netG(noise))

                # Get loss
                self.optimiserC.zero_grad()
                d_loss = (cGenerated.mean() - cReal.mean()) + gp
                d_loss.backward()
                self.optimiserC.step()

                if (i + 1) % 5 == 0:
                    # Generator
                    self.optimiserG.zero_grad()
                    noise = torch.randn(b_size, self.noiseLength, device=self.device)

                    gGenerated = self.netC(self.netG(noise))
                    g_loss = - gGenerated.mean()
                    g_loss.backward()
                    self.optimiserG.step()

            self.losses['c'].append(d_loss.data.item())
            self.losses['g'].append(g_loss.data.item())

    def wasserstein_loss(self):
        pass

    def gradient_penalty(self, real_data, generated_data):
        b_size = real_data.size()[0]

        alpha = torch.rand(b_size, 1).expand_as(real_data)

        interpolated = Variable(alpha * real_data.data + (1 - alpha) * generated_data.data, requires_grad=True)

        prob_interploated = self.netC(interpolated)

        gradients = torch_grad(outputs=prob_interploated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interploated.size()), create_graph=True,
                               retain_graph=True)[0]

        gradients = gradients.view(b_size, -1)
        self.losses['gradNorm'].append(gradients.norm(2, dim=1).mean().data.item())

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def generate(self, number):

        noise = torch.randn(number, self.noiseLength, device=self.device)
        gGenerated = self.netG(noise)
        return gGenerated


class AddDimension(nn.Module):
    """ AddDimension """
    def forward(self, x):
        return x.unsqueeze(1)


class SqueezeDimension(nn.Module):
    """ Squeeze Dimension """
    def forward(self, x):
        return x.squeeze(1)
