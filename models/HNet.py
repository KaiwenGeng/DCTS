import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from hnet.models.hnet import HNet

from layers.Embed import DataEmbedding, PositionalEmbedding
from hnet.models.config_hnet import HNetConfig, SSMConfig, AttnConfig


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.seq_len = configs.seq_len
        self.embedding_dim = configs.hnet_d_model[0]
        self.embedding = nn.Linear(configs.enc_in, self.embedding_dim)
        self.position_embedding = PositionalEmbedding(self.embedding_dim)
        arch_layout = json.loads(configs.hnet_arch_layout)
        ssm_cfg = SSMConfig(
            d_conv=configs.hnet_ssm_d_conv,
            expand=configs.hnet_ssm_expand,
            d_state=configs.hnet_ssm_d_state,
            chunk_size=configs.hnet_ssm_chunk_size,
        )
        attn_cfg = AttnConfig(
            num_heads=configs.hnet_attn_num_heads,
            rotary_emb_dim=configs.hnet_attn_rotary_emb_dim,
            window_size=configs.hnet_attn_window_size,
        )
        hnet_cfg = HNetConfig(
            arch_layout=arch_layout,
            d_model=configs.hnet_d_model,
            d_intermediate=configs.hnet_d_intermediate,
            ssm_cfg=ssm_cfg,
            attn_cfg=attn_cfg,
        )
        self.hnet = HNet(config=hnet_cfg, stage_idx=0, dtype=torch.bfloat16)

        self.out_layer = nn.Linear(self.embedding_dim, configs.c_out, bias=False)
        # asset the label len is the same as the seq_len
        assert self.label_len == self.seq_len, "Label len must be the same as the seq len"

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        assert self.label_len == self.seq_len, "Label len must be the same as the seq len"
        assert torch.equal(x_enc, x_dec[:, :self.label_len, :]), "x_enc must be the same as the first part of x_dec"
        assert torch.equal(x_mark_enc, x_mark_dec[:,:self.label_len,:]) , "x_mark_enc must be the same as the first part of x_mark_dec"
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        x_dec[:, :self.seq_len, :] = x_dec[:, :self.seq_len, :] - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc
        x_dec[:, :self.seq_len, :] = x_dec[:, :self.seq_len, :] / std_enc
        hnet_input = self.embedding(x_dec) + self.position_embedding(x_dec)
        # convert the hnet_input to bfloat16
        hnet_input = hnet_input.to(torch.bfloat16)
        B, L = x_dec.size(0), x_dec.size(1)
        mask = torch.zeros(B, L, dtype=torch.bool, device=x_dec.device)
        mask[:, :self.seq_len] = True  # True for lookback, False for padded forecast
        # make sure the hnet is in bfloat16
        self.hnet = self.hnet.to(torch.bfloat16)
        self.out_layer = self.out_layer.to(torch.bfloat16)
        hnet_output, main_network_output, boundary_predictions = self.hnet(
            hidden_states=hnet_input,
            mask=mask,
            inference_params=None,
        )
        x_out = self.out_layer(hnet_output)
        x_out = x_out * std_enc + mean_enc
        x_out = x_out.to(torch.float32)

        return x_out, boundary_predictions


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            x_out, boundary_predictions = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return x_out[:, -self.pred_len:, :], boundary_predictions

        

        # other tasks not implemented