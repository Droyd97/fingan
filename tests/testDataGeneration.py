from fingan.TestData import SineData
import pytest
import matplotlib.pyplot as plt
import numpy as np


def testSine():
    data = SineData(100, 1, [0, 10], [0, 20])
    print(data.dataset)
    plt.plot(np.linspace(0, 2 * np.pi, 100), data.dataset.T)
    plt.show()
