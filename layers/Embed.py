import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    Data Embedding Layer: Combines multiple types of embeddings
    
    Combines:
    1. Value Embedding: Converts raw time series values to d_model dimension
    2. Positional Embedding: Adds position information (sinusoidal)
    3. Temporal Embedding: Adds time features (hour, day, month, etc.)
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        # Value embedding: Convert raw features to d_model dimension
        # Input: [B, seq_len, c_in] -> Output: [B, seq_len, d_model]
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        # Positional embedding: Add position information using sinusoidal encoding
        # Input: [B, seq_len, d_model] -> Output: [B, seq_len, d_model]
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        # Temporal embedding: Add time features (hour, day, month, etc.)
        # Input: [B, seq_len, 4] -> Output: [B, seq_len, d_model]
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Forward pass through embedding layer
        
        Args:
            x: [B, seq_len, c_in] - Raw time series data
            x_mark: [B, seq_len, 4] - Time features (month, day, weekday, hour)
        Returns:
            output: [B, seq_len, d_model] - Combined embeddings
        """
        if x_mark is None:
            # No time features: only value + positional embedding
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # Full embedding: value + temporal + positional
            # This is the typical case for time series forecasting
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class MiniAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        attn_scores = self.attn(x)                 
        attn_weights = F.softmax(attn_scores, dim=1) 
        pooled = torch.sum(attn_weights * x, dim=1)  
        return pooled

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
class PatchEmbedding_Reconstruct(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout, nvars, latent_dim, reconstruction_mode):
        super(PatchEmbedding_Reconstruct, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.nvars = nvars
        self.d_model = d_model
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.vae_dropout = nn.Dropout(dropout)

        # Reconstruction mode
        self.reconstruction_mode = reconstruction_mode
        self.mini_attention_pooling = MiniAttentionPooling(d_model)
        
        
        # latent_dim
        self.latent_dim = latent_dim
        if self.reconstruction_mode == 'c_ind':
            self.latent_mixing = nn.Identity()
        elif self.reconstruction_mode == 'c_dep':
            self.latent_mixing = nn.Sequential(
                Transpose(1, 2),
                nn.Linear(self.nvars, self.nvars, bias=False),
                Transpose(1, 2)
            )
        else:
            raise ValueError(f'Reconstruction mode must be either c_dep or c_ind')
        self.reconstruction_mu = nn.Linear(d_model, self.latent_dim, bias=False)
        self.reconstruction_logvar = nn.Linear(d_model, self.latent_dim, bias=False)

    
    def forward(self, x):
        # do patching
        assert x.shape[1] == self.nvars
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        latent = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) 
        latent = self.value_embedding(latent) + self.position_embedding(latent)
        # shape here is [bs * nvars x patch_num x d_model]
        
        # # plain average pooling
        # reconstruction_latent = latent.mean(dim = 1) # [bs * nvars x d_model]


        # attention pooling
        reconstruction_latent = self.mini_attention_pooling(latent)

        reconstruction_latent = reconstruction_latent.reshape(-1, self.nvars, self.d_model) # [bs x nvars x d_model]

        reconstruction_latent = self.latent_mixing(reconstruction_latent) # [bs x nvars x d_model]

        reconstruction_latent = self.vae_dropout(reconstruction_latent)

        mu = self.reconstruction_mu(reconstruction_latent) # [bs x nvars x latent_dim]
        logvar = self.reconstruction_logvar(reconstruction_latent) # [bs x nvars x latent_dim]
        sigma = torch.exp(0.5 * logvar) # [bs x nvars x latent_dim]
        if self.training:
            sampled_latent = mu + sigma * torch.randn_like(mu) # [bs x nvars x latent_dim]
        else:
            sampled_latent = mu

        # Input encoding
        return self.dropout(latent), sampled_latent, self.nvars, mu, logvar