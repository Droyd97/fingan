import torch
import pandas as pd


class RollingWindow():
    def __init__(self, tensor):
        self.tensor = tensor

    def split(self, window_len, stride, begin=True):
        num_series = ((self.tensor.shape[1] - window_len) // stride) + 1
        output = torch.zeros(num_series, window_len, self.tensor.shape[0])
        for time_series in range(self.tensor.shape[0]):
            for idx, series_start in enumerate(range(0, self.tensor.shape[1] - window_len, stride)):
                output[idx, :, time_series] = self.tensor[time_series, series_start:series_start + window_len]
        return output


def df_to_tensor(file_dir, columns=[], header=False, transpose=True):
    df = pd.read_csv(file_dir)
    mat = df[columns].to_numpy()
    if transpose:
        mat = mat.T
    return torch.from_numpy(mat).float()


def log_returns(tensor, dim=1):
    # output = tensor.zeros(tensor.shape[0]. tensor.shape[1] - 1)
    if dim == 1:
        # for i in range(tensor.shape[1]):
        return torch.log(tensor[:, 1:] / tensor[:, :-1])
