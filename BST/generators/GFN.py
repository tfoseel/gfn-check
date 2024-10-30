import torch
import torch.nn as nn
import torch.nn.functional as F

# Function to parse the input string into tokens
def parse_sequence(input_string):
    tokens = input_string.split("->")
    return tokens

# Function to map tokens to integer indices
def encode_tokens(tokens):
    """Assigns unique integer indices to each token and encodes the sequence."""
    unique_values = sorted(set([token for token in tokens if token not in ["TRUE", "FALSE"]]))
    value_to_index = {val: i for i, val in enumerate(unique_values)}
    boolean_to_index = {"TRUE": len(unique_values), "FALSE": len(unique_values) + 1}

    # Encode each token as an integer index
    sequence_indices = [
        value_to_index[token] if token in value_to_index else boolean_to_index[token]
        for token in tokens
    ]
    
    return torch.tensor(sequence_indices), len(unique_values) + 2  # (sequence tensor, vocab size)

# Function to initialize the embedding and LSTM layers
def initialize_model(vocab_size, embedding_dim=8, hidden_size=16):
    """Initializes the embedding layer and LSTM model."""
    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    lstm_layer = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
    return embedding_layer, lstm_layer

# Function to process the input sequence through embeddings and LSTM
def process_sequence(input_string, embedding_layer, lstm_layer):
    """Processes the input sequence string through embedding and LSTM layers."""
    # Parse and encode the sequence
    tokens = parse_sequence(input_string)
    sequence_indices, vocab_size = encode_tokens(tokens)
    
    # Pass through embedding layer
    embedded_sequence = embedding_layer(sequence_indices.unsqueeze(0))  # Add batch dim

    # Pass through LSTM layer
    output, (hidden, cell) = lstm_layer(embedded_sequence)
    
    return output, hidden, cell



# Example usage
input_string = "a->TRUE->b->FALSE->c->TRUE"
tokens = parse_sequence(input_string)
sequence_tensor, vocab_size = encode_tokens(tokens)

# Initialize model components
embedding_dim = 8  # Dimension of each embedding vector
hidden_size = 16  # Hidden size of LSTM
embedding_layer, lstm_layer = initialize_model(vocab_size, embedding_dim, hidden_size)

# Process the sequence
output, hidden, cell = process_sequence(input_string, embedding_layer, lstm_layer)

print("LSTM Output Tensor:")
print(output)
print("LSTM Hidden State:")
print(hidden)
print("LSTM Cell State:")
print(cell)