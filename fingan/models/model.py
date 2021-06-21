import torch


class Model():
    def __init__(self, workers=2, batchSize=128, num_epochs=100, n_critic=5, lr=0.0002, beta1=0.5, ngpu=0):
        # Number of workers for dataloader
        self.workers = workers

        # Batch size during training
        self.batch = batchSize

        # Number of training epochs
        self.num_epochs = num_epochs

        # Number of critic iterations per iteration of the generator
        self.n_critic = n_critic

        # Number of GPUs available. Use 0 for CPU mode
        self.ngpu = ngpu

        # Set device to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        # Track Losses
        self.losses = {}

    def train():
        pass

    def generate():
        pass
