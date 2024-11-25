import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import defaultdict
import numpy as np
import itertools
import math
import random
from tqdm import tqdm

losses = []


class GFNOracle(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, domains, transformer=True):
        super(GFNOracle, self).__init__()
        self.learners = {}
        self.choice_sequence = []
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab = dict()
        self.transformer = transformer
        if transformer:
            self.hidden_dim = embedding_dim
            hidden_dim = embedding_dim
        # 1 for embedding for empty sequence, and the other is total vocabulary size
        vocab_idx = 1
        for domain, idx in domains:
            domain = list(domain)
            self.learners[idx] = GFNLearner(hidden_dim, domain)
            self.vocab[idx] = dict()
            for x in domain:
                self.vocab[idx][x] = vocab_idx
                vocab_idx += 1
        num_embeddings = 1 + sum(map(lambda d: len(d[0]), domains))
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.beta = 1
        self.logZ = nn.Parameter(torch.tensor(5.0))
        self.logZ_lower = 10
        self.lstm_pf = nn.LSTM(input_size=embedding_dim,
                               hidden_size=self.hidden_dim, batch_first=True)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=1)
        
        self.transformer_pf = nn.TransformerEncoder(transformer_layer, num_layers=10)

        self.logPf = torch.tensor(0.0)
        self.beta = 10
        self.loss = torch.tensor(0.0)
        self.num_generation = 0
        if transformer:
            self.optimizer_policy = torch.optim.Adam(
                [
                    {'params': self.embedding_layer.parameters()},
                    {'params': self.transformer_pf.parameters()},
                    {'params': itertools.chain(
                        *(learner.action_selector.parameters() for learner in self.learners.values()))},
                ],
                lr=0.001,
            )
        else:
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
            [{'params': [self.logZ], 'lr': 0.1}],
        )

    def clamp_logZ(self):
        with torch.no_grad():
            self.logZ.copy_(torch.clamp(self.logZ, min=self.logZ_lower))

    def encode_choice_sequence(self):
        return [0] + list(map(lambda x: self.vocab[x[0]][x[1]], self.choice_sequence))

    def select(self, domain, idx):
        # Get hidden state
        # print("Encoding: ", self.encode_choice_sequence())
        sequence_embeddings = self.embedding_layer(
            torch.tensor(self.encode_choice_sequence(),
                         dtype=torch.long).unsqueeze(0)
        )
        # print(sequence_embeddings)
        if self.transformer:
            hidden = self.transformer_pf(sequence_embeddings)
            hidden = hidden[:, 0, :]
        else:
            _, (hidden, _) = self.lstm_pf(sequence_embeddings)
            hidden = hidden[-1]  # shape: (1, hidden_dim)
        # Select action based on the hidden state
        choice, log_prob, probs = self.learners[idx].policy(hidden)
        # if len(self.encode_choice_sequence()) == 2:
        #     print("Choice: ", choice)
        #     print(probs)
        self.choice_sequence.append((idx, choice, log_prob))
        self.logPf = self.logPf + log_prob
        return choice

    def reward(self, reward):
        loss = (self.logPf + self.logZ -
                torch.log(torch.Tensor([reward])) * self.beta) ** 2
        tqdm.write(f"Reward: {reward} Loss: {loss.item()} Z: {math.exp(self.logZ.item())}")
        losses.append(loss.item())
        self.loss = self.loss + loss
        self.num_generation += 1
        if self.num_generation > 0 and self.num_generation % 10 == 0:
            self.optimizer_policy.zero_grad()
            self.optimizer_logZ.zero_grad()
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
            tqdm.write(str(list(map(lambda x: x[1], self.choice_sequence))))
            # print("Running mean 100: ", sum(
            #     losses[-100:]) / 100, "log Z: ", self.logZ.item())
            self.optimizer_policy.step()
            self.optimizer_logZ.step()
            self.loss = torch.tensor(0.0)
            self.clamp_logZ()

        # Reset choice sequence after updating
        self.choice_sequence = []
        self.logPf = torch.tensor(0.0)


class GFNLearner:
    def __init__(self, hidden_dim, domain):
        self.exploration_prob = 1
        self.min_exploration_prob = 0.2
        self.domain = domain
        self.action_selector = nn.Linear(
            in_features=hidden_dim, out_features=len(domain))

    def policy(self, hidden):
        if self.exploration_prob > self.min_exploration_prob:
            self.exploration_prob *= 0.9995
        output = self.action_selector(hidden)
        probs = F.softmax(output, dim=-1)  # Convert to probabilities
        # epsilon greedy
        if np.random.binomial(1, 0.5):
            sampled_index = random.choice(range(len(self.domain)))
        else:
            sampled_index = torch.multinomial(probs, 1).item()
        # sampled_index = torch.multinomial(probs, 1).item()
        return self.domain[sampled_index], torch.log(probs[0][sampled_index]), probs
