import random


class RandomOracle:
    def select(self, domain, idx):
        return random.choice(domain)

    def reward(self, r):
        return None
