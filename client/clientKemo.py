from collections import OrderedDict
import flwr as fl
import torch
from utils import *
from Network import *
class FedAvgClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, valid_loader, epoch, lossf, optimizer, DEVICE, trainF=train, validF=valid):
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
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(self.net, self.train_loader, None, self.epoch, self.lossf, self.optim, self.DEVICE, None)
        return self.get_parameters(config={}), len(self.train_loader), {}
        
    