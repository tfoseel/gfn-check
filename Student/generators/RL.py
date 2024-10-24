import random


class RLOracle:
    def select(self, domain, idx):
        raise NotImplementedError

    def reward(self, r):
        raise NotImplementedError
