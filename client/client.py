from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from Old_train import train, valid
import warnings
import os
from torch.optim import SGD
import numpy as np
import random
class FedAvgClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, valid_loader, epoch, lossf, optimizer, DEVICE, trainF=None, validF=None):
        super(FedAvgClient, self).__init__()
        self.net = net
        self.keys = net.state_dict().keys()
        self.train_loader = train_loader
        self.epoch = epoch
        self.lossf = lossf
        self.optim = optimizer
        self.DEVICE=DEVICE
        self.valid_loader= valid_loader
        self.train = trainF
        self.valid = validF
    def get_parameters(self, config):
        return [val.cpu().detach().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(self.net, self.train_loader, None, self.epoch, self.lossf, self.optim, self.DEVICE, None)
        return self.get_parameters(config={}), len(self.train_loader), {}
        
    # def evaluate(self, parameters, config):
        # self.set_parameters(parameters)
        # history = self.valid(self.net, self.valid_loader, None, lossf=self.lossf, DEVICE=self.DEVICE)
        # return history["loss"], len(self.valid_loader), {key:value for key, value in history.items() if key != "loss" }
        # return 1.0, 0, {"accuracy":0.95}

    