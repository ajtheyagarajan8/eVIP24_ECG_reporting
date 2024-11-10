import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# As implemented by Implemented by group 'Prna' physionet challenge 2020 
# https://www.cinc.org/archives/2020/pdf/CinC2020-107.pdf
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):   
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add position encodings to embeddings
        # x: embedding vects, [B x L x d_model]
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)



class Transformer(nn.Module):
    '''
    Transformer encoder processes convolved ECG samples
    Stacks a number of TransformerEncoderLayers

    Implemented by group 'Prna' physionet challenge 2020 
    https://www.cinc.org/archives/2020/pdf/CinC2020-107.pdf

    '''
    def __init__(self, d_model: int, h: int, d_ff: int, num_layers:int, dropout):
        """
        Args:
            d_model (int): size of latent feature space used in transformer
            h (int): number of attention heads in Transformer
            d_ff (int): num neurons in hidden layer
            num_layers (int): number of transformational layers 
            dropout (float) : prevent overfitting
        """
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.pe = PositionalEncoding(d_model, dropout=0.1)
        
        encode_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.h, 
            dim_feedforward=self.d_ff, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encode_layer, self.num_layers)
            
    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.pe(out)
        out = out.permute(1, 0, 2)
        out = self.transformer_encoder(out)
        out = out.mean(0) # global pooling
        return out


class CTN(nn.Module):
    # d_model = 256
    def __init__(self, d_model:int, dropout_rate, deepfeat_sz, classes):
        """
        Modified implementation of physionet challenge 2020 model from group 'Prna'
        (https://www.cinc.org/archives/2020/pdf/CinC2020-107.pdf)

        Implement classification based only the 'Deep Features' of model (excluding demographic info/metadata) 
        Requiring only waveform data as input (no demographic/metadata on input)

        Hybrid CNN and Transformer Architecture

        Output size dependent on number of classes
        
        Args:
            d_model: size of latent feature space used in transformer
            dropout_rate: dropout_probability - use 0.2
            deepfeat_sz: size of feature vector outputted by fc1 - 64
            classes: Classes of Arrhythmia - not sure the size of classification here
        """
        super(CTN, self).__init__()
        

        self.encoder = nn.Sequential( # downsampling factor = 20
            nn.Conv1d(12, 128, kernel_size=14, stride=3, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=14, stride=3, padding=0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )

        self.transformer = Transformer(d_model, h=8, d_ff=2048, num_layers=8, dropout=0.2)
    
        self.fc1 = nn.Linear(d_model, deepfeat_sz)
        self.fc2 = nn.Linear(deepfeat_sz, len(classes))
        self.dropout = nn.Dropout(dropout_rate)
            
        def _weights_init(m):
            """
            Apply Kaiming Initialisation to layers with ReLU
            """

            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        #self.apply(_weights_init)

    
            
    def forward(self, x):
        z = self.encoder(x)          # encoded sequence is batch_sz x nb_ch x seq_len
        out = self.transformer(z)    # transformer output is batch_sz x d_model
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(torch.cat([out], dim=1))
        return out                