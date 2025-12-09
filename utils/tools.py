import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visual_boundary(gt, pred, boundary_mask, name='./pic/test.pdf', boundary_prob=None):
    """
    Results visualization
    Here we also need to visualize the boundary mask
    The boundary mask is a binary mask of the same length as the input data
    We add vertical dotted lines to indicate the boundary/chunk positions
    """
    
    if boundary_prob is not None:
        # Create two subplots: main plot and probability bar chart below
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True)
        
        # Top plot: time series
        ax1.plot(pred, label='Prediction', linewidth=2)
        ax1.plot(gt, label='GroundTruth', linewidth=2)
        boundary_positions = np.where(boundary_mask)[0]
        for pos in boundary_positions:
            ax1.axvline(x=pos, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
        ax1.legend()
        ax1.set_ylabel('Value')
        
        # Bottom plot: boundary probability bar chart
        is_boundary_prob = boundary_prob[:,1]
        x_positions = np.arange(len(is_boundary_prob))
        # bar chart, but make the place where less than 0.5 orange where it is greater than 0.5 purple
        colors = ['orange' if p <= 0.5 else 'purple' for p in is_boundary_prob]
        ax2.bar(x_positions, is_boundary_prob, color=colors, alpha=0.6, width=1.0) 
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Boundary Prob')
        ax2.set_ylim([0, 1])
        
    else:
        # Single plot if no boundary_prob
        plt.figure(figsize=(12, 6))
        plt.plot(pred, label='Prediction', linewidth=2)
        plt.plot(gt, label='GroundTruth', linewidth=2)
        boundary_positions = np.where(boundary_mask)[0]
        for pos in boundary_positions:
            plt.axvline(x=pos, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
        plt.legend()
        plt.xlabel('Time Step')
        plt.ylabel('Value')
    
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
