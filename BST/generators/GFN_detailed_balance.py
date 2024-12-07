import torch.nn.functional as F
import torch.nn as nn
import torch
import itertools
from tqdm import tqdm
import math

losses = []


class GFNOracle_detailed_balance(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, domains, transformer=True):
        super(GFNOracle_detailed_balance, self).__init__()
        self.learners = {}
        self.choice_sequence = []
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab = dict()
        self.transformer = transformer
        self.prev_flow = 0.0
        self.prev_curr = []
        if transformer:
            self.hidden_dim = embedding_dim
            hidden_dim = embedding_dim

        # Initialize vocabulary and learners
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

        # Model parameters
        self.beta = 1
        self.lstm_pf = nn.LSTM(input_size=embedding_dim,
                               hidden_size=self.hidden_dim, batch_first=True)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=1)
        self.transformer_pf = nn.TransformerEncoder(transformer_layer, num_layers=10)

        self.loss = torch.tensor(0.0)
        self.num_generation = 0
        # Initialize detailed balance loss tensor
        self.detailed_balance_loss = torch.zeros(1, requires_grad=True)

        # Optimizers
        self.optimizer_policy = torch.optim.Adam(
            [
                {'params': self.embedding_layer.parameters()},
                {'params': self.transformer_pf.parameters()},
                {'params': itertools.chain(
                    *(learner.action_selector.parameters() for learner in self.learners.values()))},
            ],
            lr=1,  # Consider lowering this for stability
        )

    def encode_choice_sequence(self):
        """Encodes the current choice sequence into embeddings."""
        return [0] + list(map(lambda x: self.vocab[x[0]][x[1]], self.choice_sequence))

    def select(self, learner_idx):
        """Selects an action for a given domain and index."""
        sequence_embeddings = self.embedding_layer(
            torch.tensor(self.encode_choice_sequence(), dtype=torch.long).unsqueeze(0)
        )
        if self.transformer:
            hidden = self.transformer_pf(sequence_embeddings)
            hidden = hidden[:, 0, :]
        else:
            _, (hidden, _) = self.lstm_pf(sequence_embeddings)
            hidden = hidden[-1]  # shape: (1, hidden_dim)

        decision_idx, domain, flows = self.learners[learner_idx].policy(hidden)
        if len(self.encode_choice_sequence()) == 2:
            tqdm.write(f"Flows: {str(flows)}")

        # Track flows
        self.prev_curr.append((self.prev_flow, flows.sum()))
        self.prev_flow = flows[decision_idx].item()
        self.choice_sequence.append((learner_idx, domain[decision_idx]))
        return domain[decision_idx]

    def reward(self, reward):
        reward = reward

        # Calculate loss
        # For all steps except the first:
        # - Intermediate steps: (prev_in - curr_out)^2
        # - Terminal step: (prev_in - reward)^2
        # prev_in is the incoming flow at that step, curr_out is the outgoing flow
        loss = torch.tensor(0.0, requires_grad=True)
        for idx, (prev, curr) in enumerate(self.prev_curr):
            if idx == 0:
                # No incoming flow to compare at the very first step
                continue
            if idx == len(self.prev_curr) - 1:
                # Last step is terminal: match incoming flow to reward
                step_loss = (torch.tensor([prev], dtype=torch.float, requires_grad=True) -
                             torch.tensor([reward], dtype=torch.float, requires_grad=True)) ** 2
            else:
                # Intermediate step: match incoming flow and outgoing flow
                step_loss = (torch.tensor([prev], dtype=torch.float, requires_grad=True) -
                             torch.tensor([curr], dtype=torch.float, requires_grad=True)) ** 2

            loss = loss + step_loss

        tqdm.write(f"Detailed balance loss: {loss.item()}")
        self.num_generation += 1

        # Accumulate loss into self.detailed_balance_loss
        # We don't use += to avoid in-place operations on a leaf tensor
        self.detailed_balance_loss = self.detailed_balance_loss + loss

        # Every 10 generations, update parameters
        if self.num_generation > 0 and self.num_generation % 10 == 0:
            self.optimizer_policy.zero_grad()
            self.detailed_balance_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
            self.optimizer_policy.step()
            # Reinitialize detailed_balance_loss for next iterations
            self.detailed_balance_loss = torch.zeros(1, requires_grad=True)

        # Reset states after updating
        self.choice_sequence = []
        self.prev_curr = []
        self.prev_flow = 0.0  # Reset prev_flow to 0

class GFNLearner:
    def __init__(self, hidden_dim, domain):
        self.min_exploration_prob = 0.2
        self.domain = domain
        self.action_selector = nn.Linear(
            in_features=hidden_dim, out_features=len(domain))

    def policy(self, hidden):
        flows = F.softplus(self.action_selector(hidden)[0])
        probs = F.softmax(flows, dim=-1)  # Convert to probabilities
        sampled_index = torch.multinomial(probs, 1).item()
        return sampled_index, self.domain, flows