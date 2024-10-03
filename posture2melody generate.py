import torch
import torch.nn as nn
import numpy as np
import librosa.display
import warnings
import mediapipe as mp
import os
import cv2
import pandas as pd
from IPython.display import Audio
import soundfile as sf

vid_dir = '/Users/liuchenshu/Documents/Research/Posture2Melody/mediadata/test.mp4'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Step1: read in video and extract landmark information
mp_drawing = mp.solutions.drawing_utils #visualizing poses using visual indicators
mp_pose = mp.solutions.pose #pose estimation models (solutions)

def Euc_dist(pts, cm, img_h, img_w):
        distances_sq = (pts - cm) ** 2
        euc_distances = distances_sq.sum(axis = 1)
        diag_length = np.sqrt(img_h**2 + img_w**2)
        return euc_distances * diag_length

def landmark_data_gen(vid_dir, landmark_dir = None):
    # suppress warning messages
    warnings.filterwarnings('ignore')

    landmarks = [f'landmark_{i}' for i in range(0, 33)]
    features = ['x', 'y', 'distance_to_cm']
    comprehensive_landmarks_df_columns = pd.MultiIndex.from_product([landmarks, features], names=['landmark', 'feature'])
    comprehensive_landmarks_df = pd.DataFrame(columns = comprehensive_landmarks_df_columns)

    cap = cv2.VideoCapture(vid_dir)
    with mp_pose.Pose(min_detection_confidence=0.5, # using the pose estimation model
                    min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read() # frame is the actual image of each frame

            if not ret: # ending the loop when the video is done playing
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False #not modifiable
            results = pose.process(image) #making the actual detection
            image.flags.writeable = True #now modifiable
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #opencv wants image file in BGR format, we need to rerender the images
            image_h, image_w, _ = image.shape
            
            if results.pose_landmarks: #sometimes cannot capture any landmarks (mediapipe requires full body view for processing)
                landmarks = results.pose_landmarks.landmark #list of pose landmarks

                # convert landmark object into dataframe
                landmarks_df = pd.DataFrame(columns=['x', 'y'])
                for idx, landmark in enumerate(landmarks):
                    new_entry = np.array([landmark.x, landmark.y])
                    landmarks_df.loc[idx,:] = new_entry

                ######## For All landmarks #########
                # for all landmark points, keep track of their relative position w.r.t to center of mass (important for posture reconstruction)
                all_cm_x, all_cm_y = landmarks_df['x'].mean(), landmarks_df['y'].mean()
                landmarks_df['distance_to_cm'] = Euc_dist(np.array(landmarks_df.loc[:, ['x', 'y']]), 
                                                        np.array([all_cm_x, all_cm_y]),
                                                        image_h, image_w)

                ######## Storing all important features #########
                # x, y coordinates, distance of each landmark from the center of mass
                new_data = pd.DataFrame(columns=comprehensive_landmarks_df_columns)
                for landmark_idx in range(33):
                    landmark_name = f"landmark_{landmark_idx}"
                    for feature_idx in range(len(features)):
                        feature_name = features[feature_idx]
                        new_data[(landmark_name, feature_name)] = [landmarks_df.loc[:, feature_name].iloc[landmark_idx]]
                comprehensive_landmarks_df = pd.concat([comprehensive_landmarks_df, new_data], ignore_index=True)

                # finding and visualizing center of mass
                cm_x, cm_y = landmarks_df['x'].mean(), landmarks_df['y'].mean()
                cm_x_coord, cm_y_coord = int(cm_x * image_w), int(cm_y * image_h)
                cv2.circle(img = image, 
                        center = (cm_x_coord, cm_y_coord), 
                        radius=5, 
                        color = (0, 0, 255), 
                        thickness = -1)

            else:
                pass #when landmarks are not detected

            mp_drawing.draw_landmarks(image = image, 
                                    landmark_list = results.pose_landmarks, #coordinate of each landmark detected by the pose estimation model
                                    connections = mp_pose.POSE_CONNECTIONS, #the connections of each pose landmark
                                    landmark_drawing_spec = mp_drawing.DrawingSpec(color = (245, 117, 66), thickness=2, circle_radius=2),
                                    connection_drawing_spec = mp_drawing.DrawingSpec(color = (245, 117, 66), thickness=2, circle_radius=2))

            cv2.imshow('Webcam Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'): #0xFF is checking which key is being pressed
                break

        if landmark_dir is not None: np.save(landmark_dir, comprehensive_landmarks_df)
        cap.release() #release the use of webcame
        cv2.destroyAllWindows() #close all cv2 associated windows
        cv2.waitKey(1) #need this line!! else the mediapipe window maynot close properly for running on local vscode (basically giving the kernel enough time to close the windows)

    # Re-enable warnings
    warnings.filterwarnings('default')
    return comprehensive_landmarks_df

input_landmark_data = landmark_data_gen(vid_dir=vid_dir)
#print(input_landmark_data.head)
input_landmark_data_tensor = torch.tensor(input_landmark_data.values, dtype=torch.float32).unsqueeze(0).to(device) #[1, num_frames, n_features]
print(f"Input tensor has shape: {input_landmark_data_tensor.shape}")

# Step2: process landmark source and generate mel-spectrogram target (only need the transformer as generator)
hidden_dim = 32
nhead = 4
num_layers = 3
num_source_features = 99
num_mel_bins = 128
num_epochs = 50
batch_size = 3 # ERROR: batch_size larger than 4 will return nan loss (RuntimeError: all elements of input should be between 0 and 1)
generator_lr = 1e-7
discriminator_ls = 1e-2
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

def pad_sequences_to_max_length(sequence, max_len, padding_value=0):
    """
    Pad each sequence in the input list to the specified max_len.

    Returns:
        torch.Tensor: Padded sequences tensor of shape (batch_size, max_len, feature_dim).
        torch.Tensor: Padding mask tensor of shape (batch_size, max_len).
    """
    # Pad each sequence to max_len
    seq_len = sequence.size(1)  # Length of the sequence
    feature_dim = sequence.size(2)  # Number of features

    # Create the padding to reach max_len
    if seq_len < max_len:
        # Create padding of shape (max_len - seq_len, feature_dim)
        pad = torch.full((1, max_len - seq_len, feature_dim), padding_value)
        padded_seq = torch.cat([sequence, pad], dim=1)
    else:
        # Truncate if necessary (optional, for safety)
        padded_seq = sequence[:max_len]
    
    # Create the padding mask (True for padded positions, False for real data)
    padding_mask = torch.cat([torch.zeros(seq_len), torch.ones(max_len - seq_len)]).bool()

    return padded_seq, padding_mask.unsqueeze(0) # padding_mask should have shape [batch_size, seq_len]

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
        print(f"tgt in decoder has shape {tgt.shape}")
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)
        print(f"tgt_emb has shape {tgt_emb.shape}")
        #tgt_emb = tgt_emb.transpose(0, 1)
        print(f"tgt_emb within the decoder has shape {tgt_emb.shape}")
        print(f"memory within the decoder has shape {memory.transpose(0, 1).shape}")
        output = self.transformer_decoder(tgt_emb, memory.transpose(0, 1),
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask,
                                          tgt_mask=tgt_mask)
        print(f"output shape in decoder is {output.shape}")
        output = self.fc(output)
        print(f"output shape in decoder is {output.shape}")
        return output

class TransformerModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(TransformerModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask)
        return output

def generate_square_subsequent_mask(size):
    """
    Generate a mask for self-attention in the decoder.
    Creates a triangular matrix with the segment on top of diagnal is true, below is false
    """
    mask = torch.tril(torch.ones(size, size)).bool()  # Upper triangular mask
    return mask

def generate_sequence(transformer_model, src, max_length=3, start_token=None, end_token=None, device='cpu'):
    """
    Generate a sequence using a transformer model in evaluation mode.
    
    Args:
        transformer_model (nn.Module): The trained transformer encoder-decoder model.
        src (Tensor): The source input sequence, shape [src_seq_length, num_features] or [batch_size, src_seq_length, num_features].
        max_length (int): Maximum length of the generated sequence.
        start_token (Tensor): Initial start token for the target sequence, shape [1, num_features].
        end_token (int or Tensor): The token that signifies the end of the sequence.

    Returns:
        generated_sequence (Tensor): The generated sequence.
    """
    # Set model to evaluation mode
    transformer_model.eval()
    
    # Move source to the same device as the model
    src = src.to(device)
    padded_src, src_key_padded_mask = pad_sequences_to_max_length(src, max_len = 2000)
    #print(f"the padded_src has type: {padded_src.shape}") # tuple
    print(f"padding mask is {src_key_padded_mask.shape}")

    # Encode the source sequence
    memory = transformer_model.encoder(padded_src, src_key_padded_mask)
    print(f"shape of memory is {memory.shape}")

    # Initialize the target sequence with the start token
    if start_token is None:
        # Create a tensor of zeros as the initial token if no start token is provided
        tgt = torch.zeros(1, num_mel_bins).unsqueeze(0).to(device)  # Shape [1, 1, num_features]
        tgt_mask = torch.triu(torch.ones((1, 1)) * float('-inf'), diagonal=1)
    else:
        tgt = start_token.unsqueeze(0).to(device)  # Shape [1, 1, num_features]

    # Prepare a list to store generated tokens
    generated_sequence = [tgt]
    #tgt_mask = torch.zeros((2000, 2000), dtype = torch.bool)

    for _ in range(max_length):
        # Run the transformer decoder with the current target sequence and encoded memory
        #padded_tgt, tgt_key_padding_mask = pad_sequences_to_max_length(tgt, max_len = 2000)
        #tgt_key_padding_mask = None
        output = transformer_model.decoder(tgt, memory, 
                                           tgt_key_padding_mask = None,
                                           memory_key_padding_mask = src_key_padded_mask,
                                           tgt_mask=tgt_mask)

        # Get the last generated token (the last output position)
        #print(f"the output is {output}!!!!!!")
        #print(f"the shape of the output is {output.shape}")
        next_token = output[-1, :, :].unsqueeze(0)
        #print(f"the next token is {output[-1, :, :]}")
        #print(f"shape of next_token is {next_token.shape}")
        
        # Append the next token to the generated sequence
        generated_sequence.append(next_token)
        #print(f"tgt has shape: {tgt.shape}")

        # Update the target sequence by appending the new token
        tgt = torch.cat((tgt, next_token), dim=0)  # Append the token to the current target sequence
        print(f"tgt after concatenation is {tgt}")
        #print(f"tgt dimension 0 is {tgt.size(0)}")
        #print(f"shape of tgt is {tgt.shape}")
        tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(device)
        tgt_mask = torch.triu(torch.ones((tgt.size(0), tgt.size(0))) * float('-inf'), diagonal=1)
        #print(f"shape of tgt_mask is {tgt_mask.shape}")
        #print(f"tgt_maks is {tgt_mask}")
        
        # Check if the generated token is the end-of-sequence token (if specified)
        if end_token is not None and (next_token == end_token).all():
            break

    # Concatenate the list of generated tokens into a single tensor
    generated_sequence = torch.cat(generated_sequence, dim=0)  # Shape [generated_length, num_features]
    
    return generated_sequence

encoder = TransformerEncoder(hidden_dim=hidden_dim, nhead=nhead, num_layers=num_layers, n_features=num_source_features)
decoder = TransformerDecoder(hidden_dim=hidden_dim, nhead=nhead, num_layers=num_layers, mel_dim=num_mel_bins)
transformer_generator = TransformerModel(encoder, decoder)
checkpoint = torch.load('./model/transformer_discriminator_epoch10.pth')
transformer_generator.load_state_dict(checkpoint['transformer_state_dict'])
#transformer_generator.eval()
with torch.no_grad():
    #output = transformer_generator(input_landmark_data)
    output = generate_sequence(transformer_generator, input_landmark_data_tensor, device=device).transpose(0, 1).squeeze(0)
    print(output.shape)

# Step3: convert mel-spectogram data to audio
def mel_to_audio(mel_spectrogram, sr = 22050, n_fft = 2048, n_iter = 1):
    '''
    mel_spectrogram: [num_segments, mel_bins]
    sr: sampling rate
    n_fft: the length of segments to be fourier transformed
    n_iter: number of iterations to run
    '''
    if isinstance(mel_spectrogram, torch.Tensor): # sometimes input are torch tensors
        mel_spectrogram = mel_spectrogram.cpu().numpy()

    # Inverse Mel-spectrogram to STFT
    stft_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram.T, sr=sr, n_fft=n_fft)
    # Reconstruct the waveform using Griffin-Lim algorithm
    waveform = librosa.griffinlim(stft_spectrogram, n_iter=n_iter)

    return waveform

waveform = mel_to_audio(output)
sf.write('reconstructed_audio.wav', waveform, samplerate=22050)
Audio(waveform, rate=22050)