import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class PaddedDataset(Dataset): # unifying across the entire dataset
    def __init__(self, input_file_dir, output_file_dir):
        # Load the input and output datasets
        self.input_file_dir = input_file_dir
        self.output_file_dir = output_file_dir
        self.inputs = []
        for input_dir in self.input_file_dir:
            self.inputs.append(np.load(input_dir))
        self.outputs = []
        for output_dir in self.output_file_dir:
            self.outputs.append(np.load(output_dir))

        # Calculate the maximum length across the whole dataset for inputs and outputs (NUCLEAR option!!!)
        self.max_input_len = max(inp.shape[0] for inp in self.inputs)
        self.max_output_len = max(out.shape[0] for out in self.outputs)

    def __len__(self):
        return len(self.input_file_dir)

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.inputs[idx], dtype=torch.float32)
        output_seq = torch.tensor(self.outputs[idx], dtype=torch.float32)

        # Pad the input sequence to max_input_len
        padded_input = torch.cat([input_seq, torch.zeros(self.max_input_len - input_seq.size(0), input_seq.size(1))], dim=0)

        # Create input mask
        input_mask = (padded_input.sum(dim=1) != 0)  # Shape: (max_input_len, )

        # Pad the output sequence to max_output_len
        padded_output = torch.cat([output_seq, torch.zeros(self.max_output_len - output_seq.size(0), output_seq.size(1))], dim=0)

        # Create output mask
        output_mask = (padded_output.sum(dim=1) != 0)  # Shape: (max_output_len, )

        return {
            'input': padded_input,
            'target': padded_output,
            'input_mask': input_mask,
            'target_mask': output_mask
        }

main_dir = './mediadata'  
landmark_paths = [os.path.join(main_dir, 'landmark', landmark_name) for landmark_name in os.listdir(os.path.join(main_dir, 'landmark')) if landmark_name != '.DS_Store']
mel_paths = [os.path.join(main_dir, 'mel_spectrogram', landmark_name) for landmark_name in os.listdir(os.path.join(main_dir, 'mel_spectrogram')) if landmark_name != '.DS_Store']
dataset = PaddedDataset(landmark_paths, mel_paths)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=3)

# Step 2: Define the Transformer Encoder-Decoder Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, nhead, num_layers, n_features):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(n_features, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, src, src_key_padding_mask=None):
        src_emb = self.embedding(src)
        src_emb = self.pos_encoder(src_emb)
        src_emb = src_emb.transpose(0, 1)  # Shape: [n_frame, batch_size, hidden_dim]
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory.transpose(0, 1)  # Shape: [batch_size, n_frame, hidden_dim]

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, nhead, num_layers, mel_dim):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(mel_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, mel_dim)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=None):
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # Shape: [n_frame, batch_size, hidden_dim]
        output = self.transformer_decoder(tgt_emb, memory.transpose(0, 1),
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask,
                                          tgt_mask=tgt_mask)
        output = self.fc(output)
        return output.transpose(0, 1)  # Shape: [batch_size, n_frame, mel_dim]

class TransformerModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(TransformerModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask)
        return output

# Define model parameters
hidden_dim = 32
nhead = 4
num_layers = 3
num_source_features = 99
num_mel_bins = 128

encoder = TransformerEncoder(hidden_dim, nhead, num_layers, num_source_features)
decoder = TransformerDecoder(hidden_dim, nhead, num_layers, num_mel_bins)
model = TransformerModel(encoder, decoder)

# Step 3: Training the Model
class StableMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(StableMSELoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        # Calculate the MSE loss with added stability
        loss = (input - target) ** 2
        # Adding epsilon to prevent NaN when computing mean or sum
        return torch.mean(loss + self.eps)

# Usage
criterion = StableMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10 
model.train()  # Set model to training mode

for epoch in range(num_epochs):
    total_loss = 0  # Variable to accumulate loss

    for batch in dataloader:
        # Get padded inputs and targets
        input_seq = batch['input'].to(device)  # Shape: (batch_size, max_input_len, input_dim)
        target_seq = batch['target'].to(device)  # Shape: (batch_size, max_output_len, output_dim)
        input_mask = batch['input_mask'].to(device)  # Input mask
        target_mask = batch['target_mask'].to(device)  # Output mask
        input_padding_mask = ~input_mask
        '''
        The tgt_key_padding_mask in the Transformer model is used to indicate which positions in the target sequence 
        are padding and should be ignored during attention computations. 
        True (1) indicate the position should NOT be attended to
        False (0) indicate the position should be attended to
        '''
        target_padding_mask = ~target_mask

        tgt_mask = torch.triu(torch.ones((target_seq.shape[1], target_seq.shape[1])) * float('-inf'), diagonal=1)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float(0.0))
        assert torch.isfinite(tgt_mask).any(dim=-1).all(), "Each row in tgt_mask should have at least one non-masked position."
        print(torch.any(torch.isnan(tgt_mask)), "Check for NaNs in tgt_mask")

        # Zero the gradients
        optimizer.zero_grad()

        output = model(input_seq, target_seq, 
                       src_key_padding_mask=input_padding_mask, 
                       tgt_key_padding_mask=target_padding_mask, 
                       memory_key_padding_mask=input_padding_mask,
                       tgt_mask = tgt_mask)

        # Calculate loss
        loss = criterion(output, target_seq)  # Modify if needed based on your output processing
        # print(f"loss is {loss}")
        # print(torch.isinf(loss))
        total_loss += loss.item()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        break

    # Print average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')