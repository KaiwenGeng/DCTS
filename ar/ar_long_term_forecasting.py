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
        # assert self.args.pred_len == 1, "Pred len must be 1 for autoregressive training"


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





    
    def autoregressive_inference(self, seq, seq_mark):
        predictions = []
        current_input = seq.clone()  # Shape: (B, seq_len, features)
        
        for i in range(self.args.pred_len):
            if self.model_name in self.dc_models:
                outputs, boundary_predictions = self.model(current_input, seq_mark)
            else:
                outputs = self.model(current_input, seq_mark)
            
            # Get the next step prediction
            next_pred = outputs[:, -1:, :]  # Shape: (B, 1, features)
            predictions.append(next_pred)
            
            # Slide window: drop first step, append prediction
            current_input = torch.cat([current_input[:, 1:, :], next_pred], dim=1)
        
        return torch.cat(predictions, dim=1) 

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
            for i, (batch_x, batch_seq, batch_x_mark, batch_seq_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                SEQ_LABEL = batch_seq[:, self.args.seq_len:, :].float().to(self.device)
                batch_seq = batch_seq[:, :self.args.seq_len + 1, :]
                # print(f"the shape of batch_seq: {batch_seq.shape}")
                # # make sure the batch_seq is just batch x with 1 extra step
                # print(f"check the value matching: {torch.equal(batch_seq[:, :-1, :], batch_x)}")
                batch_x = batch_x.float().to(self.device)
                batch_seq = batch_seq.float().to(self.device) # [batch_size, label_len + pred_len, num_features], [x0, x1, x2, ..., x95, 96] if our seq len is 96 (because we always "1 step" ahead of the input sequence)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_seq_mark = batch_seq_mark.float().to(self.device)
                if self.model_name in self.dc_models:
                    outputs, boundary_predictions = self.model(batch_x, batch_x_mark) # takes in [x0, x1, x2, ..., x95] and outputs [x1_pred, x2_pred, ..., x96_pred]
                else:
                    outputs = self.model(batch_x, batch_x_mark) # takes in [x0, x1, x2, ..., x95] and outputs [x1_pred, x2_pred, ..., x96_pred]
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
                # train_loss.append(loss.item())
                with torch.no_grad():
                    seq_pred = self.autoregressive_inference(batch_x, batch_x_mark)
                # print(f"the shape of seq_pred: {seq_pred.shape}")
                # print(f"the shape of SEQ_LABEL: {SEQ_LABEL.shape}")
                # breakpoint()
                seq_loss = criterion(seq_pred, SEQ_LABEL)
                train_loss.append(seq_loss.item())

                if (i + 1) % 1 == 0:
                    print("\titers: {0}, epoch: {1} | seq loss: {2:.7f}".format(i + 1, epoch + 1, seq_loss.item()))
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            ratio_loss = np.average(ratio_loss)
            breakpoint()


                
