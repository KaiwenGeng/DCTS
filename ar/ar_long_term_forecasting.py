from data_provider.data_factory import data_provider
from ar.ar_basic import AR_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_boundary
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from hnet.utils.train import load_balancing_loss
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class AR_Long_Term_Forecast(AR_Basic):
    def __init__(self, args):
        super(AR_Long_Term_Forecast, self).__init__(args)
        self.model_name = args.model
        # make sure the label_len is 0
        assert self.args.label_len == self.args.seq_len, "Label len must be the same as the seq len for autoregressive training"
        self.single_batch = True


    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def generate(self, batch_x, pred_len, single_batch=True):
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module

        # 1. Normalization (matching ARDC.py forecast logic)
        mean_enc = batch_x.mean(1, keepdim=True).detach()
        std_enc = torch.sqrt(torch.var(batch_x - mean_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        batch_x_norm = (batch_x - mean_enc) / std_enc

        B, L, C = batch_x.shape
        device = batch_x.device
        
        predictions_list = []

        # Iterate over each sample in the batch to handle batch_size=1 restriction in HNet inference
        for i in range(B):
            # Shape: [1, L, C]
            sample_x = batch_x_norm[i:i+1]
            
            # 2. Allocate Inference Cache for single sample
            inference_params = model.hnet.allocate_inference_cache(1, L + pred_len, dtype=torch.bfloat16)
            
            # 3. Prefill (Process Prompt)
            # Embed and convert to bfloat16 as expected by HNet
            hidden_states = model.embedding(sample_x).to(torch.bfloat16)
            mask = torch.ones(1, L, dtype=torch.bool, device=device)
            
            with torch.inference_mode():
                # Pass prompt through HNet to fill cache
                # Returns: hidden_states, main_network_output, boundary_predictions
                hidden_states_out, _, _ = model.hnet(
                    hidden_states,
                    mask=mask,
                    inference_params=inference_params
                )
                
                # Use the last hidden state to predict the first future step
                last_hidden = hidden_states_out[:, -1:, :] # [1, 1, D]
                
                # Project to output space
                model.out_layer = model.out_layer.to(torch.bfloat16)
                next_token_norm = model.out_layer(last_hidden) # [1, 1, C]
                
                sample_generated = []
                sample_generated.append(next_token_norm)
                
                curr_token_norm = next_token_norm
                
                # 4. Generation Loop
                for _ in range(pred_len - 1):
                    # Embed current prediction to get input for next step
                    # Ensure input matches embedding layer dtype
                    curr_input = curr_token_norm.to(device).to(model.embedding.weight.dtype)
                    curr_embed = model.embedding(curr_input).to(torch.bfloat16) # [1, 1, D]
                    
                    # Step through HNet
                    hidden_states_step, _, _ = model.hnet.step(curr_embed, inference_params)
                    
                    # Project
                    curr_token_norm = model.out_layer(hidden_states_step)
                    sample_generated.append(curr_token_norm)
            
            # Concatenate time steps for this sample: [1, pred_len, C]
            predictions_list.append(torch.cat(sample_generated, dim=1))
            if single_batch:
                break

        # 5. Combine batch
        predictions = torch.cat(predictions_list, dim=0) # [B, pred_len, C] or [1, pred_len, C] if single_batch
        
        # 6. Denormalize - slice stats if single_batch
        if single_batch:
            predictions = predictions.float() * std_enc[:1] + mean_enc[:1]
        else:
            predictions = predictions.float() * std_enc + mean_enc
        
        return predictions

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            ratio_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (_, batch_seq, _, batch_seq_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                SEQ_LABEL = batch_seq[:, self.args.seq_len:, :].float().to(self.device)
                # batch_seq = batch_seq[:, :self.args.seq_len + 1, :]
                model_input = batch_seq[:, :-1, :]
                # print(f"the shape of batch_seq: {batch_seq.shape}")
                # # make sure the batch_seq is just batch x with 1 extra step
                # print(f"check the value matching: {torch.equal(batch_seq[:, :-1, :], batch_x)}")
                model_input = model_input.float().to(self.device)
                batch_seq = batch_seq.float().to(self.device) # [batch_size, label_len + pred_len, num_features], [x0, x1, x2, ..., x95, 96] if our seq len is 96 (because we always "1 step" ahead of the input sequence)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                batch_seq_mark = batch_seq_mark.float().to(self.device)
                if self.model_name in self.dc_models:
                    outputs, boundary_predictions = self.model(model_input, None) # takes in [x0, x1, x2, ..., x95] and outputs [x1_pred, x2_pred, ..., x96_pred]
                else:
                    outputs = self.model(model_input, None) # takes in [x0, x1, x2, ..., x95] and outputs [x1_pred, x2_pred, ..., x96_pred]
                # print(f"the shape of last step's prediction: {outputs[:,-1,:].shape}")
                # print(f"the shape of last step's true value: {batch_seq[:, -1, :].shape}")
                # breakpoint()
                tf_training_loss = criterion(outputs, batch_seq[:, 1:, :]) # since we asserted that the pred len is 1, we can just take the last step's prediction and the last step's true value. Teacher forcing training loss.
                if self.model_name in self.dc_models:
                    '''
                    HNet Specific
                    '''
                    moe_loss = 0.0
                    for obj in boundary_predictions:
                        moe_loss += self.args.hnet_moe_loss_weight * load_balancing_loss(obj, self.args.hnet_num_experts)
                    joint_loss = tf_training_loss + moe_loss
                    ratio_loss.append(moe_loss.item())
                train_loss.append(tf_training_loss.item())

                if (i + 1) % 20 == 0:
                    with torch.no_grad():
                        
                        generated_output = self.generate(model_input[:, :self.args.seq_len, :], pred_len=self.args.pred_len, single_batch=self.single_batch)
                        if self.single_batch:
                            seq_loss = criterion(generated_output, SEQ_LABEL[:1])
                        else:
                            seq_loss = criterion(generated_output, SEQ_LABEL)
                    print("\titers: {0}, epoch: {1} | tf loss: {2:.7f} | moe loss: {3:.7f} | seq loss: {4:.7f}".format(i + 1, epoch + 1, tf_training_loss.item(), moe_loss.item(), seq_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.model_name in self.dc_models:
                    joint_loss.backward()
                else:
                    tf_training_loss.backward()
                model_optim.step()
            breakpoint()


                    


                