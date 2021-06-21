import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torch.autograd import grad as torch_grad
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm


class QuantGAN():
    """ QuantGAN """
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

        class TemporalBlock(nn.Module):
            def __init__(self, input, hidden, output):
                self.input = input
                self.hidden = hidden
                self.output = output
                self.block = nn.Sequential(
                    spectral_norm(NN.Conv1d(self.input, self.hidden, padding=3), n_power_iterations=10),
                    nn.PReLU(),
                    spectral_norm(NN.Conv1d(self.hidden, self.output, padding=3), n_power_iterations=10),
                    nn.PReLU()
                )

            def forward(self, x):
                return self.block(x)

        class Generator(nn.Module):
            def __init__(self, ngpu):
                self.ngpu = ngpu
                self.gen_architecture = nn.Sequential(
                    TemporalBlock(),

                )