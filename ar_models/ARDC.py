import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from hnet.models.hnet import HNet

from layers.Embed import PositionalEmbedding
from hnet.models.config_hnet import HNetConfig, SSMConfig, AttnConfig


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.embedding_dim = configs.hnet_d_model[0]
        self.embedding = nn.Linear(configs.enc_in, self.embedding_dim)
        self.position_embedding = PositionalEmbedding(self.embedding_dim)
        arch_layout = json.loads(configs.hnet_arch_layout)
        self.c_in = configs.enc_in
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

    def forecast(self, seq, seq_mark):
        mean_enc = seq.mean(1, keepdim=True).detach()
        seq = seq - mean_enc
        std_enc = torch.sqrt(torch.var(seq, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        seq = seq / std_enc
        # print(f"the shape of seq: {seq.shape}")
        hnet_input = self.embedding(seq) + self.position_embedding(seq)
        hnet_input = hnet_input.to(torch.bfloat16)
        self.out_layer = self.out_layer.to(torch.bfloat16)
        # print(f"the shape of hnet_input: {hnet_input.shape}")


        mask = torch.zeros(seq.shape[0], seq.shape[1], dtype=torch.bool, device=seq.device)
        mask[:, :] = True  
        # mask = mask.repeat_interleave(self.c_in, dim=0) # no padding so all True
        # print(f"the shape of mask: {mask.shape}")
        # breakpoint()

        # make sure the hnet is in bfloat16
        self.hnet = self.hnet.to(torch.bfloat16)
        hnet_output, main_network_output, boundary_predictions = self.hnet(
            hidden_states=hnet_input,
            mask=mask,
            inference_params=None,
        )
        x_out = self.out_layer(hnet_output)

        x_out = x_out * std_enc + mean_enc
        x_out = x_out.to(torch.float32)

        return x_out, boundary_predictions


    def forward(self, seq, seq_mark, mask=None):
        if self.task_name == 'ar_long_term_forecast':
            x_out, boundary_predictions = self.forecast(seq, seq_mark)
            return x_out[:, :, :], boundary_predictions

        
