import random
import math
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        num_embeddings = 1 + sum(map(lambda d: len(d[0]), domains))
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm_pf = nn.LSTM(input_size=embedding_dim,
                               hidden_size=self.hidden_dim, batch_first=True)

        # Optimizer for embedding, logZ, and logPb
        self.logZ = nn.Parameter(torch.tensor(0.0))
        self.logPf = nn.Parameter(torch.tensor(0.0))
        self.optimizer = torch.optim.Adam(
            list(self.embedding_layer.parameters()) +
            list(self.lstm_pf.parameters()) +
            [self.logZ, self.logPf], lr=0.001)

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
                torch.log(torch.Tensor([reward]))) ** 2
        print(loss)
        self.optimizer.zero_grad()
        for learner in self.learners.values():
            learner.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for learner in self.learners.values():
            learner.optimizer.step()

        # Reset choice sequence after updating
        self.choice_sequence = []
        self.logPf = nn.Parameter(torch.tensor(0.0))


class GFNLearner:
    def __init__(self, hidden_dim, domain):
        self.domain = domain
        self.action_selector = nn.Linear(
            in_features=hidden_dim, out_features=len(domain))
        self.optimizer = torch.optim.Adam(
            self.action_selector.parameters(), lr=0.001)

    # def reward(self, loss, oracle_optimizer):
    #     # Perform backpropagation using the calculated loss in GFNOracle
    #     oracle_optimizer.zero_grad()
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     oracle_optimizer.step()
    #     self.optimizer.step()

    def policy(self, hidden):
        output = self.action_selector(hidden)
        probs = F.softmax(output, dim=-1)  # Convert to probabilities
        sampled_index = torch.multinomial(probs, 1).item()
        return self.domain[sampled_index], torch.log(probs[0][sampled_index])
