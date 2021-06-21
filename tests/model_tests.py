import pytest
from fingan.wgan import WGAN
from fingan.TestData import SineData
import matplotlib.pyplot as plt


def testWGAN():
    data = SineData(100, 400, [3, 10], [1, 5])
    model = WGAN()
    model.train(data, 20)
    x = model.generate(1)
    print(x.detach().numpy())
    plt.plot(x.detach().numpy()[0])
    plt.plot(data.dataset[0])
    plt.show()
    # plt.plot(model.losses['g'], label='g')
    # plt.plot(model.losses['c'], label='c')
    # plt.legend()
    plt.show()
    
