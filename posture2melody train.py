import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define model parameters
hidden_dim = 32
nhead = 4
num_layers = 3
num_source_features = 99
num_mel_bins = 128
num_epochs = 50
batch_size = 4 # ERROR: batch_size larger than 4 will return nan loss (RuntimeError: all elements of input should be between 0 and 1)
generator_lr = 1e-7
discriminator_ls = 1e-2

# Step 1: Loading Data
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
dataloader = DataLoader(dataset, batch_size=batch_size)

# Step 2: ransformer Encoder-Decoder & Discriminator Model
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

class CNNDiscriminator(nn.Module):
    def __init__(self, num_features = 128, num_filters=64, kernel_size=3, stride=1, hidden_dim=128, num_classes=1, dropout=0.3):
        """
        Args:
            num_features (int): Number of features at each time step in the sequence.
            num_filters (int): Number of filters for the convolutional layers.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride size for the convolution.
            hidden_dim (int): Number of units in the fully connected layer.
            num_classes (int): Number of output classes (1 for real/fake).
            dropout (float): Dropout probability.
        """
        super(CNNDiscriminator, self).__init__()

        # Convolutional layers to extract temporal features
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=num_filters, kernel_size=kernel_size, stride=stride, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, stride=stride, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=kernel_size, stride=stride, padding=1)

        # Fully connected layer for classification
        self.fc1 = nn.Linear(num_filters * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()  # Sigmoid for binary classification (real/fake)

    def forward(self, x):
        """
        input: x (torch.Tensor): Input tensor of shape [batch_size, num_frames, num_features].
        """
        # Reshape the input for 1D convolutions: [batch_size, num_features, num_frames]
        x = x.permute(0, 2, 1)

        # Apply convolutional layers with activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Global average pooling: [batch_size, num_filters*4, 1] -> [batch_size, num_filters*4]
        x = torch.mean(x, dim=2)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)  # Output probability

encoder = TransformerEncoder(hidden_dim, nhead, num_layers, num_source_features)
decoder = TransformerDecoder(hidden_dim, nhead, num_layers, num_mel_bins)
model = TransformerModel(encoder, decoder)
discriminator = CNNDiscriminator()

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

generator_criterion = StableMSELoss()
discriminator_criterion = nn.BCELoss()
generator_optimizer = optim.Adam(model.parameters(), lr=generator_lr)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_ls, betas=(0.5, 0.999))

model.train()  # Set model to training mode
discriminator.train()

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
        generator_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()

        output = model(input_seq, target_seq, 
                       src_key_padding_mask=input_padding_mask, 
                       tgt_key_padding_mask=target_padding_mask, 
                       memory_key_padding_mask=input_padding_mask,
                       tgt_mask = tgt_mask)
    
        # Transformer-generated content, typically of shape [batch_size, seq_len, feature_dim]
        generated_content = output  # Replace with actual transformer output
        # Ground truth sequence, typically of shape [batch_size, seq_len, feature_dim]
        groundtruth_content = target_seq  # Replace with actual ground truth data

        # Pass through discriminator
        real_prob = discriminator(groundtruth_content)  # Should be close to 1
        fake_prob = discriminator(generated_content.detach())  # Should be close to 0

        real_labels = torch.ones(real_prob.size(0), 1).to(device)  # Real labels = 1
        fake_labels = torch.zeros(fake_prob.size(0), 1).to(device)  # Fake labels = 0

        # Calculate loss
        generator_loss = generator_criterion(output, target_seq)
        real_loss = discriminator_criterion(real_prob, real_labels)
        fake_loss = discriminator_criterion(fake_prob, fake_labels)
        discriminator_loss = (real_loss + fake_loss) / 2
        # print(f"loss is {loss}")
        # print(torch.isinf(loss))
        total_loss += generator_loss.item() + discriminator_loss.item()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Backward pass and optimization
        generator_loss.backward()
        discriminator_loss.backward()
        generator_optimizer.step()
        discriminator_optimizer.step()
        break

    # Print average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    model_dir = './model'
    if (epoch + 1) % 5 == 0: # saving every 5 epochs
        model_save_dir = os.path.join(model_dir, f"transformer_discriminator_epoch{epoch+1}.pth")
        torch.save({
            'transformer_state_dict': model.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'transformer_optimizer_state_dict': generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict()
        }, model_save_dir)