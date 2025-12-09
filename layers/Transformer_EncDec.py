import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer: Self-Attention + Feed Forward + Residual Connections
    
    Architecture:
    1. Multi-Head Self-Attention
    2. Add & Norm (Residual + LayerNorm)
    3. Feed Forward Network (2 linear layers with activation)
    4. Add & Norm (Residual + LayerNorm)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # Feed forward dimension = 4 * d_model
        self.attention = attention  # Multi-head self-attention
        # Feed Forward Network: 2 linear layers
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Forward pass through encoder layer
        
        Args:
            x: [B, seq_len, d_model] - Input sequence
        Returns:
            output: [B, seq_len, d_model] - Processed sequence
            attn: Attention weights
        """
        # ========== STEP 1: MULTI-HEAD SELF-ATTENTION ==========
        # Self-attention: Q=K=V=x (attend to all positions in sequence)
        new_x, attn = self.attention(
            x, x, x,  # Query, Key, Value all from same input
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        # Residual connection + Dropout
        x = x + self.dropout(new_x)

        # ========== STEP 2: FEED FORWARD NETWORK ==========
        # First normalize, then apply feed forward
        y = x = self.norm1(x)
        # FFN: Linear -> Activation -> Linear
        # Transpose for conv1d: [B, seq_len, d_model] -> [B, d_model, seq_len]
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # Transpose back

        # ========== STEP 3: RESIDUAL CONNECTION + LAYER NORM ==========
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer: Self-Attention + Cross-Attention + Feed Forward + Residual Connections
    
    Architecture:
    1. Multi-Head Self-Attention (attend to decoder sequence)
    2. Add & Norm (Residual + LayerNorm)
    3. Multi-Head Cross-Attention (attend to encoder output)
    4. Add & Norm (Residual + LayerNorm)
    5. Feed Forward Network (2 linear layers with activation)
    6. Add & Norm (Residual + LayerNorm)
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # Feed forward dimension = 4 * d_model
        self.self_attention = self_attention  # Self-attention for decoder sequence
        self.cross_attention = cross_attention  # Cross-attention to encoder output
        # Feed Forward Network: 2 linear layers
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Layer Normalization (3 norms for 3 sub-layers)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        Forward pass through decoder layer
        
        Args:
            x: [B, label_len+pred_len, d_model] - Decoder input sequence
            cross: [B, seq_len, d_model] - Encoder output (from encoder)
        Returns:
            output: [B, label_len+pred_len, d_model] - Processed decoder sequence
        """
        # ========== STEP 1: MULTI-HEAD SELF-ATTENTION ==========
        # Self-attention: Q=K=V=x (attend to decoder sequence)
        # This allows decoder to attend to its own previous positions
        x = x + self.dropout(self.self_attention(
            x, x, x,  # Query, Key, Value all from decoder input
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # ========== STEP 2: MULTI-HEAD CROSS-ATTENTION ==========
        # Cross-attention: Q=x, K=V=cross (attend to encoder output)
        # This is the KEY mechanism: decoder attends to encoder's representation
        # Q comes from decoder, K&V come from encoder output
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,  # Query from decoder, Key&Value from encoder
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        # ========== STEP 3: FEED FORWARD NETWORK ==========
        # Same as encoder: Linear -> Activation -> Linear
        y = x = self.norm2(x)
        # FFN: Linear -> Activation -> Linear
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # ========== STEP 4: RESIDUAL CONNECTION + LAYER NORM ==========
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
