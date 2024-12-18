import random


class RandomOracle:
    def select(self, domain, idx=None):
        return random.choice(domain)

    def reward(self, r):
        return None
