import torch
from torch.utils.data import Dataset
import pandas as pd


class TimeSeries(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx, :]


# df = pd.read_csv(self.dataset_dir)
#         return df.to_numpy()