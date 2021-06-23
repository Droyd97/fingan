import torch
import logging


class Model():
    def __init__(self, name, wkdir, is_logging, workers, batch_size, num_epochs, save_point, ngpu):

        self.name = name

        # Working directory to save and load models
        self.wkdir = wkdir

        # Set up logger
        self.is_logging = is_logging
        self.setup_logging(self.is_logging)

        # Number of workers for dataloader
        self.workers = workers

        # Batch size during training
        self.batch_size = batch_size

        # Number of training epochs
        self.num_epochs = num_epochs

        # Number of epochs to save model
        self.save_point = save_point

        # Number of critic iterations per iteration of the generator
        # self.n_critic = n_critic

        # Number of GPUs available. Use 0 for CPU mode
        self.ngpu = ngpu

        # Set device to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        # Track Losses
        # self.losses = {'g': [], 'c': [], 'gp': [], 'gradNorm': [], 'iter': 0}

    def train(self):
        pass

    def generate(self, number):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def summary(self):
        pass

    def setup_logging(self, is_logging):
        if is_logging:
            if self.wkdir is None:
                self.wkdir = os.getcwd()
            log_file = self.wkdir + "/" + self.name + ".log"
            logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
