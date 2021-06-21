from torch.utils.data import Dataset
import numpy as np


class SineData(Dataset):
    def __init__(self, N, numSeries, freqRange, ampRange, random=True, seed=None):
        self.N = N
        self.numSeries = numSeries
        self.freqRange = freqRange
        self.ampRange = ampRange
        self.random = random
        self.seed = seed
        self.dataset = self.generate_data()

    def __len__(self):
        return self.numSeries

    def __getitem__(self, idx):
        return self.dataset[idx]

    def generate_data(self):
        if self.random is False:
            np.random.seed(self.seed)

        x = np.linspace(0, 2 * np.pi, num=self.N)
        lowFreq, highFreq = self.freqRange[0], self.freqRange[1]
        lowAmp, highAmp = self.ampRange[0], self.ampRange[1]

        frequencies = (highFreq - lowFreq) * np.random.rand(self.numSeries, 1) + lowFreq
        amplitudes = (highAmp - lowAmp) * np.random.rand(self.numSeries, 1) + lowAmp

        return amplitudes * np.sin(frequencies * x)
