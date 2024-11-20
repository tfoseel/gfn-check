import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import defaultdict
import numpy as np
import itertools
import math
import random

losses = []


class GFNOracle:
    def __init__(self, embedding_dim, hidden_dim, domains):
        self.learners = {}
        self.choice_sequence = []
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab = dict()
        # 1 for embedding for empty sequence, and the other is total vocabulary size
        vocab_idx = 1
        for domain, idx in domains:
            domain = list(domain)
            self.learners[idx] = GFNLearner(hidden_dim, domain)
            for x in domain:
                self.vocab[x] = vocab_idx
                vocab_idx += 1
        print(domains)
        print(self.vocab)
        num_embeddings = 1 + sum(map(lambda d: len(d[0]), domains))
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.logZ = nn.Parameter(torch.tensor(5.0))
        self.logZ_lower = 0.0
        self.lstm_pf = nn.LSTM(input_size=embedding_dim,
                               hidden_size=self.hidden_dim, batch_first=True)

        self.logPf = torch.tensor(0.0)

        self.beta = 1

        self.loss = torch.tensor(0.0)
        self.num_generation = 0
        self.optimizer_policy = torch.optim.Adam(
            [
                {'params': self.embedding_layer.parameters()},
                {'params': self.lstm_pf.parameters()},
                {'params': itertools.chain(
                    *(learner.action_selector.parameters() for learner in self.learners.values()))},
            ],
            lr=0.001,
        )
        self.optimizer_logZ = torch.optim.Adam(
            [{'params': [self.logZ], 'lr': 0.01}],
        )

    def clamp_logZ(self):
        self.logZ.data = torch.clamp(self.logZ, min=self.logZ_lower)

    def encode_choice_sequence(self):
        return [0] + list(map(lambda x: self.vocab[x[0]], self.choice_sequence))

    def select(self, domain, idx):
        # Get hidden state
        sequence_embeddings = self.embedding_layer(
            torch.tensor(self.encode_choice_sequence(),
                         dtype=torch.long).unsqueeze(0)
        )
        _, (hidden, _) = self.lstm_pf(sequence_embeddings)

        hidden = hidden[-1]  # shape: (1, hidden_dim)
        # Select action based on the hidden state
        choice, log_prob = self.learners[idx].policy(hidden)
        self.choice_sequence.append((choice, log_prob))
        self.logPf = self.logPf + log_prob
        return choice

    def reward(self, reward):
        loss = (self.logPf + self.logZ -
                torch.log(torch.Tensor([reward])) * self.beta) ** 2
        losses.append(loss.item())
        # if len(losses) > 100:
        #     print(
        #         f"Running mean 100: {sum(losses[-100:]) / 100}, choices: {list(map(lambda x: x[0], self.choice_sequence))}")
       # print("Generated tree: ", list(map(lambda x: x[0], self.choice_sequence)))
        self.loss = self.loss + loss
        self.num_generation += 1
        if self.num_generation > 0 and self.num_generation % 10 == 0:
            self.optimizer_policy.zero_grad()
            self.optimizer_logZ.zero_grad()
            self.loss.backward()
            print("Running mean 100: ", sum(losses[-100:]) / 100)
            self.optimizer_policy.step()
            self.optimizer_logZ.step()
            self.loss = torch.tensor(0.0)
            self.clamp_logZ()

        # Reset choice sequence after updating
        self.choice_sequence = []
        self.logPf = torch.tensor(0.0)


class GFNLearner:
    def __init__(self, hidden_dim, domain):
        self.domain = domain
        self.action_selector = nn.Linear(
            in_features=hidden_dim, out_features=len(domain))

    def policy(self, hidden):
        output = self.action_selector(hidden)
        probs = F.softmax(output, dim=-1)  # Convert to probabilities
        # epsilon greedy
        if np.random.binomial(1, 0.25):
            sampled_index = random.choice(range(len(self.domain)))
        else:
            sampled_index = torch.multinomial(probs, 1).item()
        return self.domain[sampled_index], torch.log(probs[0][sampled_index])
