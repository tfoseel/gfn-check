import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

# Function to parse the input string into tokens


def parse_sequence(input_string):
    tokens = input_string.split("->")
    return tokens

# Function to map tokens to integer indices


def encode_tokens(tokens):
    """Assigns unique integer indices to each token and encodes the sequence."""
    # unique_values = sorted(set([token for token in tokens if token not in ["TRUE", "FALSE"]]))
    # value_to_index = {val: i for i, val in enumerate(unique_values)}

    boolean_to_index = {"TRUE": 11, "FALSE": 12}

    # Encode each token as an integer index
    import pdb
    pdb.set_trace()
    sequence_indices = [
        token if token not in ["TRUE, FALSE"] else boolean_to_index[token]
        for token in tokens
    ]

    return torch.tensor(sequence_indices), 13  # (sequence tensor, vocab size)

    # return torch.tensor(sequence_indices), len(unique_values) + 2  # (sequence tensor, vocab size)

# Function to initialize the embedding and LSTM layers


def initialize_model(vocab_size, embedding_dim=8, num_actions=16):
    """Initializes the embedding layer and LSTM model."""
    embedding_layer = nn.Embedding(
        num_embeddings=vocab_size, embedding_dim=embedding_dim)
    lstm_layer = nn.LSTM(input_size=embedding_dim,
                         hidden_size=num_actions, batch_first=True)
    return embedding_layer, lstm_layer

# Function to process the input sequence through embeddings and LSTM


def process_sequence(input_string, embedding_layer, lstm_layer):
    """Processes the input sequence string through embedding and LSTM layers."""
    # Parse and encode the sequence
    tokens = parse_sequence(input_string)
    sequence_indices, vocab_size = encode_tokens(tokens)

    # Pass through embedding layer
    embedded_sequence = embedding_layer(
        sequence_indices.unsqueeze(0))  # Add batch dim

    # Pass through LSTM layer
    output, (hidden, cell) = lstm_layer(embedded_sequence)

    # get the last output
    output = output.squeeze(0)[-1]

    # apply softmax to get the action values
    output = F.softmax(output, dim=0)

    return output, hidden, cell


# Example usage
input_string = "2->TRUE->8->FALSE->3->TRUE"
tokens = parse_sequence(input_string)
sequence_tensor, vocab_size = encode_tokens(tokens)

# Initialize model components
embedding_dim = 8  # Dimension of each embedding vector
hidden_size = 16  # Hidden size of LSTM
embedding_layer, lstm_layer = initialize_model(
    vocab_size, embedding_dim, hidden_size)

# Process the sequence
output, hidden, cell = process_sequence(
    input_string, embedding_layer, lstm_layer)

print("LSTM Output Tensor:")
print(output.sum())
print("LSTM Hidden State:")
print(hidden)
print("LSTM Cell State:")
print(cell)


class GFNOracle:
    def __init__(self, abstract_state_fn, epsilon=0.25, gamma=1.0, initial_val=0):
        self.abstract_state_fn = abstract_state_fn
        self.learners = {}
        self.choice_sequence = []
        self.epsilon = epsilon
        self.gamma = gamma
        self.initial_val = initial_val
        self.logZ = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.lr_logz = 0.01
        self.optimizer_logZ = torch.optim.Adam(
            [
                {'params': [self.logZ], 'lr': self.lr_logz},
            ]
        )

    def select(self, domain, idx):
        abstract_state = self.abstract_state_fn(self.choice_sequence)
        if not idx in self.learners:
            self.learners[idx] = GFNLearner(
                self.epsilon, self.gamma, self.initial_val, embedding_dim=8, hidden_size=len(domain[idx]))
        choice, log_pf = self.learners[idx].policy(domain, abstract_state)
        self.choice_sequence.append(choice)
        return choice, log_pf

    # updates upon full episodes
    def reward(self, reward):
        for learner in self.learners.values():
            learner.reward(reward)
        self.choice_sequence = []


class GFNLearner:
    def __init__(self,  epsilon=0.25, gamma=1.0, learning_rate=0.01, initial_val=0, embedding_dim=8, hidden_size=16):
        self.embedding_layer_pf = nn.Embedding(
            num_embeddings=1, embedding_dim=embedding_dim)
        self.embedding_layer_pb = nn.Embedding(
            num_embeddings=1, embedding_dim=embedding_dim)
        self.lstm_layer_pf = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.lstm_layer_pb = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)

        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.choice_state_sequence = []
        self.initial_val = initial_val
        self.optimizer = torch.optim.Adam(
            [
                {'params': self.embedding_layer_pf.parameters(), 'lr': self.lr},
                {'params': self.lstm_layer_pf.parameters(), 'lr': self.lr},
            ]
        )

    def policy(self, domain, state):
        domain = list(domain)
        # Epsilon-greedy strategy
        if np.random.binomial(1, self.epsilon):
            choice = random.choice(domain)
        else:
            self.action_values = process_sequence(
                state, self.embedding_layer_pf, self.lstm_layer_pf)
            action_idx = random.choice(np.flatnonzero(
                self.action_values == self.action_values.max()))  # break ties randomly
            choice = domain[action_idx]
            log_pf = math.log(self.action_values[action_idx])
        self.choice_state_sequence.append([state, choice, 0])
        return choice, log_pf
    

        