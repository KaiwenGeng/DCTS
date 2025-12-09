import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from einops import rearrange
from layers.Embed import PositionalEmbedding, DataEmbedding_inverted
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.dc_tools import RoutingModule, ChunkLayer

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

def rearrange_boundary_predictions(boundary_predictions, c_in):
    for i in range(len(boundary_predictions)):
        boundary_predictions[i].boundary_mask = rearrange(boundary_predictions[i].boundary_mask, '(b c) l -> b c l', c=c_in)
        boundary_predictions[i].boundary_prob = rearrange(boundary_predictions[i].boundary_prob, '(b c) l d -> b c l d', c=c_in, d=2)
        boundary_predictions[i].selected_probs = rearrange(boundary_predictions[i].selected_probs, '(b c) l d -> b c l d', c=c_in, d=1)
    return boundary_predictions

class PaddingMask:
    def __init__(self, mask):
        self.mask = mask


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class EncoderLayer(nn.Module):
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
        new_x, attn = self.attention(
            x, x, x, 
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))  
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
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

    def __init__(self, time_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model 
        self.time_attention = time_attention  # Self-attention for time_wise_features
        self.cross_attention = cross_attention  # Cross-attention 
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu


    def forward(self, variate_wise_features, time_wise_features, c_in, d_model, x_mask=None, cross_mask=None, tau=None, delta=None, chunking_mask=None):
        # variate_wise_features = self.norm1(variate_wise_features)

        if chunking_mask is not None:
            # 1. Time-wise Mask (Self Attention)
            # chunking_mask is (Batch*C, L). True = Valid.
            # We need True = Padding (Invalid) for FullAttention.
            padding_mask = ~chunking_mask
            time_mask_tensor = padding_mask.unsqueeze(1).unsqueeze(1)
            time_mask_obj = PaddingMask(time_mask_tensor)
            
            # 2. Cross-wise Mask (Cross Attention)
            # Reshape from (Batch*C, L) -> (Batch, C*L)
            cross_padding_mask = rearrange(padding_mask, '(b c) l -> b (c l)', c=c_in)
            cross_mask_tensor = cross_padding_mask.unsqueeze(1).unsqueeze(1)
            cross_mask_obj = PaddingMask(cross_mask_tensor)
        else:
            # Fallback if no chunking mask provided (though expected)
            time_mask_obj = x_mask
            cross_mask_obj = cross_mask



        time_wise_features = time_wise_features + self.dropout(self.time_attention(
            time_wise_features, time_wise_features, time_wise_features,  
            attn_mask=time_mask_obj,
            tau=tau, delta=None
        )[0])
        
        time_wise_features = self.norm1(time_wise_features)

        time_wise_features_res = time_wise_features.clone()


        time_wise_features = rearrange(time_wise_features, '(b c) l d -> b (c l) d', c=c_in, d=d_model)
        variate_wise_features = variate_wise_features + self.dropout(self.cross_attention(
            variate_wise_features, time_wise_features, time_wise_features,  
            attn_mask=cross_mask_obj,
            tau=tau, delta=delta
        )[0])

        y = variate_wise_features = self.norm2(variate_wise_features)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))


        z = time_wise_features_res 
        z = self.dropout(self.activation(self.conv3(z.transpose(-1, 1))))
        z = self.dropout(self.conv4(z).transpose(-1, 1))




        return self.norm4(variate_wise_features + y), self.norm5(time_wise_features_res + z)


class Decoder(nn.Module):
    def __init__(self, c_in, d_model, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None, chunking_mask=None):
        for layer in self.layers:

            res, cross = layer(x, cross, self.c_in, self.d_model, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta, chunking_mask=chunking_mask)

        if self.norm is not None:
            res = self.norm(res)

        if self.projection is not None:
            res = self.projection(res)
        return res



# class ByteEmbedding(nn.Module):
#     def __init__(self, d_model):
#         super(ByteEmbedding, self).__init__()
#         self.d_model = d_model
#         self.embed1 = nn.Embedding(256, d_model //4)
#         self.embed2 = nn.Embedding(256, d_model //4)
#         self.embed3 = nn.Embedding(256, d_model //4)
#         self.embed4 = nn.Embedding(256, d_model //4)
#         # self.flatten = nn.Flatten(start_dim=-2)
#     def forward(self, x):
#         # make sure x is float32
#         assert x.dtype == torch.float32, "x must be float32"
#         # print("before embedding, the shape of x is", x.shape)
#         x = x.contiguous().view(torch.uint8).reshape(x.shape + (4,))
#         x = x.long()

#         result = torch.concat([self.embed1(x[:,:,0]), self.embed2(x[:,:,1]), self.embed3(x[:,:,2]), self.embed4(x[:,:,3])], dim=-1)
#         # print("result.shape:", result.shape)
#         # breakpoint()
#         # result = self.flatten(result)
#         return result



class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.seq_len = configs.seq_len
        self.c_in = configs.enc_in
        self.d_model = configs.d_model



        self.value_embedding = nn.Linear(1, self.d_model)

        self.routing = RoutingModule(d_model=self.d_model)
        self.chunk_layer = ChunkLayer()

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)  #
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )



        self.decoder = Decoder(
            c_in=self.c_in, d_model=self.d_model,
            layers=[
                    DecoderLayer(
                        # Self-attention: Attend to decoder input sequence
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        # Cross-attention: Attend to encoder output
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)  # d_layers=1 decoder layer
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
            )


        self.global_embedding = nn.Linear(self.seq_len, self.d_model //2)
        self.variate_identity = nn.Parameter(torch.randn(1, self.c_in, self.d_model //2))
        self.global_norm = nn.LayerNorm(self.d_model)

        self.position_embedding = PositionalEmbedding(self.d_model)

        self.output_head = nn.Linear(self.d_model, self.pred_len)

        self.revin_layer = RevIN(num_features=self.c_in)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        assert self.label_len == self.seq_len, "Label len must be the same as the seq len"
        assert torch.equal(x_enc, x_dec[:, :self.label_len, :]), "x_enc must be the same as the first part of x_dec"
        assert torch.equal(x_mark_enc, x_mark_dec[:,:self.label_len,:]) , "x_mark_enc must be the same as the first part of x_mark_dec"
        x_enc = self.revin_layer(x_enc, mode='norm')


        mask = torch.zeros(x_enc.shape[0], self.seq_len, dtype=torch.bool, device=x_dec.device)
        mask[:, :] = True  
        mask = mask.repeat_interleave(self.c_in, dim=0)

        x_global = self.global_embedding(x_enc.permute(0, 2, 1)) # [B, c, d_model //2]
        x_id = self.variate_identity.expand(x_enc.shape[0], -1, -1)
        x_global = torch.cat([x_global, x_id], dim=-1)

        x_global = self.global_norm(x_global)

        x_enc = rearrange(x_enc, 'b l c -> (b c) l')


        hidden_states = self.value_embedding(x_enc.unsqueeze(-1)) 

        hidden_states, _ = self.encoder(hidden_states + self.position_embedding(hidden_states))



        bpred_output = self.routing(
            hidden_states,
            mask=mask,
        )

        hidden_states, _ , _, chunking_mask = self.chunk_layer(
            hidden_states, bpred_output.boundary_mask, None, mask=mask, selected_probs=bpred_output.selected_probs
        )





        decoder_output = self.decoder(x_global, hidden_states + self.position_embedding(hidden_states), chunking_mask=chunking_mask)

        decoder_output = self.output_head(decoder_output)

        decoder_output = decoder_output.permute(0, 2, 1)

    

        # decoder_output = decoder_output * stdev + means

        decoder_output = self.revin_layer(decoder_output, mode='denorm')

        boundary_predictions = rearrange_boundary_predictions([bpred_output], self.c_in)

        return decoder_output, boundary_predictions


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, boundary_predictions = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], boundary_predictions