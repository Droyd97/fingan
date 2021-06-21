from abc import ABC, abstractmethod


class DataGenerator(ABC):
    def __init__(self):
        super().__init__()

    def generate(self):
        pass
